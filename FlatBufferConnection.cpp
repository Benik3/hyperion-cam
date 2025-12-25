#include <iostream>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/un.h>

#include "FlatBufferConnection.h"
#include "hyperhdr_reply_generated.h"
#include "hyperhdr_request_generated.h"

FlatBufferConnection::FlatBufferConnection()
	: _sockfd(0), _registered(false), _connected(false), _skipReply(false)
{
}

FlatBufferConnection::~FlatBufferConnection()
{
	close(_sockfd);
}

int FlatBufferConnection::connectToServer(const char *origin, const char *host, uint16_t port, int priority, bool skipReply)
{
	if (_sockfd > 0) // already connected
	{
		return 1;
	}
	else
	{
		_origin = origin;
		_priority = priority;
		_connected = false;
		_registered = false;
		_sockfd = 0;
		_skipReply = skipReply;
		struct sockaddr_in serv_addr;
		struct sockaddr_un serv_unaddr;
		int domain;

		if (strcmp(host, "/tmp/hyperhdr-domain") == 0)
		{
			domain = AF_UNIX;
		}
		else
		{
			domain = AF_INET;
		}

		if ((_sockfd = socket(domain, SOCK_STREAM, 0)) == -1)
		{
			std::cerr << "ERROR: Could not create socket" << std::endl;
			_sockfd = 0;
			return 0;
		}
		struct timeval timeout;
		timeout.tv_sec = 5;
		timeout.tv_usec = 0;

		if (setsockopt(_sockfd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout,
					   sizeof(timeout)) < 0)
		{
			std::cerr << "ERROR: setsockopt SO_SNDTIMEO failed" << std::endl;
			connectionDestroy();
			return 0;
		}

		if (setsockopt(_sockfd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout,
					   sizeof(timeout)) < 0)
		{
			std::cerr << "ERROR: setsockopt SO_RCVTIMEO failed" << std::endl;
			connectionDestroy();
			return 0;
		}

		memset(&serv_addr, '0', sizeof(serv_addr));
		memset(&serv_unaddr, '0', sizeof(serv_unaddr));

		if (domain == AF_INET)
		{
			serv_addr.sin_family = AF_INET;
			int flags = 1;
			if (setsockopt(_sockfd, SOL_TCP, TCP_NODELAY, (void *)&flags, sizeof(flags)))
			{
				std::cerr << "ERROR: setsockopt TCP_NODELAY failed" << std::endl;
				connectionDestroy();
				return 0;
			}

			serv_addr.sin_port = htons(port);

			if (inet_pton(AF_INET, host, &serv_addr.sin_addr) <= 0)
			{
				std::cerr << "ERROR: Failed to parse IP address: " << host << std::endl;
				connectionDestroy();
				return 0;
			}

			if (connect(_sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
			{
				std::cerr << "ERROR: Connection Failed " << std::endl;
				connectionDestroy();
				return 0;
			}
		}
		else
		{
			serv_unaddr.sun_family = AF_UNIX;
			strcpy(serv_unaddr.sun_path, host);

			if (connect(_sockfd, (struct sockaddr *)&serv_unaddr, SUN_LEN(&serv_unaddr)) < 0)
			{
				std::cerr << "ERROR: Connection Failed " << std::endl;
				connectionDestroy();
				return 0;
			}
		}

		_connected = true;

		return 1;
	}
}

void FlatBufferConnection::connectionDestroy()
{
	_connected = false;
	_registered = false;
	if (_sockfd > 0)
		close(_sockfd);
	_sockfd = 0;
}

int FlatBufferConnection::readData()
{
	if (!_sockfd)
		return -1;

	uint8_t headbuff[4];
	int n = read(_sockfd, headbuff, 4);

	uint32_t messageSize =
		((headbuff[0] << 24) & 0xFF000000) |
		((headbuff[1] << 16) & 0x00FF0000) |
		((headbuff[2] << 8) & 0x0000FF00) |
		((headbuff[3]) & 0x000000FF);
	if (n < 0)
	{
		connectionDestroy();
		std::cerr << "ERROR: Read from socket failed " << std::endl;
		return -1;
	}
	else if (messageSize >= sizeof(recvBuff))
	{
		std::cerr << "ERROR: Response from Hyperion is too long" << std::endl;
		return -1;
	}

	n = read(_sockfd, recvBuff, messageSize);
	if (n < 0)
	{
		connectionDestroy();
		std::cerr << "ERROR: Read from socket failed" << std::endl;
		return -1;
	}
	parseReply(hyperhdrnet::GetReply(recvBuff));
	return 0;
}

int FlatBufferConnection::setRegister(const char *origin, int priority)
{
	if (!_sockfd)
		return -1;

	auto registerReq = hyperhdrnet::CreateRegister(_builder, _builder.CreateString(origin), priority);
	auto req = hyperhdrnet::CreateRequest(_builder, hyperhdrnet::Command_Register, registerReq.Union());
	_builder.Finish(req);

	uint32_t size = _builder.GetSize();
	int ret = sendMessage(_builder.GetBufferPointer(), size);
	_builder.Clear();

	if (ret != -1)
	{
		ret = readData();
	}

	return ret;
}

int FlatBufferConnection::setImage(const uint8_t *buffer, uint16_t width, uint16_t height)
{
	if (!_registered)
	{
		if (setRegister(_origin, _priority) == -1)
			return -1;
	}

	auto imgData = _builder.CreateVector(buffer, width * height * 3);
	auto rawImg = hyperhdrnet::CreateRawImage(_builder, imgData, width, height);
	auto imageReq = hyperhdrnet::CreateImage(_builder, hyperhdrnet::ImageType_RawImage, rawImg.Union(), -1);
	auto req = hyperhdrnet::CreateRequest(_builder, hyperhdrnet::Command_Image, imageReq.Union());
	_builder.Finish(req);

	int ret = sendMessage(_builder.GetBufferPointer(), _builder.GetSize());
	_builder.Clear();

	if (ret != -1 && !_skipReply)
		readData();

	return ret;
}

int FlatBufferConnection::sendMessage(const uint8_t *buffer, uint32_t size)
{
	if (!_sockfd)
		return -1;
	if (!_connected)
		return -1;

	const uint8_t header[] = {
		(uint8_t)((size >> 24) & 0xFF),
		(uint8_t)((size >> 16) & 0xFF),
		(uint8_t)((size >> 8) & 0xFF),
		(uint8_t)(size & 0xFF)};

	// write message
	if (write(_sockfd, header, 4) < 0)
	{
		connectionDestroy();
		return -1;
	}

	auto result = write(_sockfd, buffer, size);
	if (result < 0)
	{
		connectionDestroy();
		return -1;
	}

	return 0;
}

bool FlatBufferConnection::parseReply(const hyperhdrnet::Reply *reply)
{

	if (!reply->error())
	{
		// no error set must be a success or registered or video
		int32_t videoMode = reply->video();
		int32_t registered = reply->registered();
		if (videoMode != -1)
		{
			// We got a video reply.
			std::cout << "Set video mode " << videoMode << std::endl;
			return true;
		}

		// We got a registered reply.
		if (registered == _priority)
		{
			std::cout << "Client registered succesfully" << std::endl;
			_registered = true;
		}

		return true;
	}
	else
	{
		std::cerr << "ERROR: Server replied with error: " << reply->error() << std::endl;
	}

	return false;
}
