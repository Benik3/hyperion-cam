#include <iostream>
#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <poll.h>

namespace
{
	bool parseIpAddressFamily(const char *host, int &outFamily)
	{
		if (host == nullptr || *host == '\0')
			return false;

		in_addr addr4;
		if (inet_pton(AF_INET, host, &addr4) == 1)
		{
			outFamily = AF_INET;
			return true;
		}

		in6_addr addr6;
		if (inet_pton(AF_INET6, host, &addr6) == 1)
		{
			outFamily = AF_INET6;
			return true;
		}

		return false;
	}

	ssize_t readFully(int fd, void *buf, size_t len)
	{
		uint8_t *p = static_cast<uint8_t *>(buf);
		size_t done = 0;
		while (done < len)
		{
			ssize_t n = ::read(fd, p + done, len - done);
			if (n > 0)
			{
				done += static_cast<size_t>(n);
				continue;
			}
			if (n == 0)
				return 0; // EOF
			if (errno == EINTR)
				continue;
			return -1;
		}
		return static_cast<ssize_t>(done);
	}

	// Gather-write: header + payload in one syscall (iov is consumed/adjusted).
	ssize_t writevFully(int fd, struct iovec *iov, int iovcnt)
	{
		size_t total = 0;
		for (int i = 0; i < iovcnt; i++)
			total += iov[i].iov_len;

		size_t done = 0;
		while (done < total)
		{
			ssize_t n = ::writev(fd, iov, iovcnt);
			if (n < 0)
			{
				if (errno == EINTR)
					continue;
				return -1;
			}
			if (n == 0)
				return -1;
			done += static_cast<size_t>(n);

			size_t adv = static_cast<size_t>(n);
			while (adv > 0 && iovcnt > 0)
			{
				if (adv >= iov[0].iov_len)
				{
					adv -= iov[0].iov_len;
					iov++;
					iovcnt--;
				}
				else
				{
					iov[0].iov_base = static_cast<uint8_t *>(iov[0].iov_base) + adv;
					iov[0].iov_len -= adv;
					adv = 0;
				}
			}
		}
		return static_cast<ssize_t>(done);
	}
} // namespace

#include "FlatBufferConnection.h"
#include "hyperhdr_reply_generated.h"
#include "hyperhdr_request_generated.h"

FlatBufferConnection::FlatBufferConnection()
	: _sockfd(0), _registered(false), _connected(false), _skipReply(false), _connectFailedReported(false)
{
}

FlatBufferConnection::~FlatBufferConnection()
{
	if (_sockfd > 0)
		close(_sockfd);
}

void FlatBufferConnection::reserveBuilderBytes(size_t bytes)
{
	// Grow-only. Allocate a throwaway uninitialized vector to force the internal
	// buffer to the needed capacity, then Clear() (which keeps the allocation).
	// Note: PreAlign(bytes, 1) does NOT grow the buffer - it only writes the
	// alignment padding, which is 0 bytes for alignment 1.
	if (bytes <= _reservedBuilderBytes)
		return;

	_builder.Clear();
	uint8_t *scratch = nullptr;
	_builder.CreateUninitializedVector(bytes, &scratch);
	_builder.Clear();
	_reservedBuilderBytes = bytes;
}

void FlatBufferConnection::setConnectTimeoutMs(int timeoutMs)
{
	// Clamp to a sane range; 0 means "no wait".
	if (timeoutMs < 0)
		timeoutMs = 0;
	if (timeoutMs > 60000)
		timeoutMs = 60000;
	_connectTimeoutMs = timeoutMs;
}

void FlatBufferConnection::setSendTimeoutMs(int timeoutMs)
{
	// Clamp to a sane range; keep a small floor so a transient hiccup doesn't
	// abandon every frame, but bound it well below the old 5s default.
	if (timeoutMs < 10)
		timeoutMs = 10;
	if (timeoutMs > 60000)
		timeoutMs = 60000;
	_sendTimeoutMs = timeoutMs;
}

static bool connectWithTimeout(int fd, const struct sockaddr *addr, socklen_t addrlen, int timeoutMs)
{
	const int oldFlags = fcntl(fd, F_GETFL, 0);
	if (oldFlags < 0)
		return false;

	if (fcntl(fd, F_SETFL, oldFlags | O_NONBLOCK) < 0)
		return false;

	int rc = ::connect(fd, addr, addrlen);
	if (rc == 0)
	{
		(void)fcntl(fd, F_SETFL, oldFlags);
		return true;
	}

	if (errno != EINPROGRESS)
	{
		(void)fcntl(fd, F_SETFL, oldFlags);
		return false;
	}

	struct pollfd pfd;
	pfd.fd = fd;
	pfd.events = POLLOUT;
	pfd.revents = 0;

	rc = ::poll(&pfd, 1, timeoutMs);
	if (rc <= 0)
	{
		// timeout or error
		(void)fcntl(fd, F_SETFL, oldFlags);
		if (rc == 0)
			errno = ETIMEDOUT;
		return false;
	}

	int soErr = 0;
	socklen_t soErrLen = sizeof(soErr);
	if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &soErr, &soErrLen) < 0)
	{
		(void)fcntl(fd, F_SETFL, oldFlags);
		return false;
	}

	(void)fcntl(fd, F_SETFL, oldFlags);
	if (soErr != 0)
	{
		errno = soErr;
		return false;
	}

	return true;
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
		struct sockaddr_in6 serv_addr6;
		struct sockaddr_un serv_unaddr;
		int domain = AF_UNIX;
		int ipFamily = AF_UNIX;
		if (parseIpAddressFamily(host, ipFamily))
			domain = ipFamily;

		if ((_sockfd = socket(domain, SOCK_STREAM, 0)) == -1)
		{
			std::cerr << "ERROR: Could not create socket" << std::endl;
			_sockfd = 0;
			return 0;
		}
		// Send timeout bounds how long a blocking write may stall on a slow/hung
		// server (latency-first: abandon the frame fast). Receive timeout stays
		// generous because reads only happen during one-time registration.
		struct timeval sndTimeout;
		sndTimeout.tv_sec = _sendTimeoutMs / 1000;
		sndTimeout.tv_usec = (_sendTimeoutMs % 1000) * 1000;

		struct timeval timeout;
		timeout.tv_sec = 5;
		timeout.tv_usec = 0;

		if (setsockopt(_sockfd, SOL_SOCKET, SO_SNDTIMEO, (char *)&sndTimeout,
					   sizeof(sndTimeout)) < 0)
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
		memset(&serv_addr6, '0', sizeof(serv_addr6));
		memset(&serv_unaddr, '0', sizeof(serv_unaddr));

		if (domain == AF_INET || domain == AF_INET6)
		{
			int flags = 1;
			if (setsockopt(_sockfd, SOL_TCP, TCP_NODELAY, (void *)&flags, sizeof(flags)))
			{
				std::cerr << "ERROR: setsockopt TCP_NODELAY failed" << std::endl;
				connectionDestroy();
				return 0;
			}

			if (domain == AF_INET)
			{
				serv_addr.sin_family = AF_INET;
				serv_addr.sin_port = htons(port);
				if (inet_pton(AF_INET, host, &serv_addr.sin_addr) != 1)
				{
					std::cerr << "ERROR: Failed to parse IPv4 address: " << host << std::endl;
					connectionDestroy();
					return 0;
				}
				if (!connectWithTimeout(_sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr), _connectTimeoutMs))
				{
					if (!_connectFailedReported)
					{
						std::cerr << "ERROR: Connection Failed";
						if (errno)
							std::cerr << ": " << std::strerror(errno);
						std::cerr << "\n";
						_connectFailedReported = true;
					}
					connectionDestroy();
					return 0;
				}
			}
			else
			{
				serv_addr6.sin6_family = AF_INET6;
				serv_addr6.sin6_port = htons(port);
				if (inet_pton(AF_INET6, host, &serv_addr6.sin6_addr) != 1)
				{
					std::cerr << "ERROR: Failed to parse IPv6 address: " << host << std::endl;
					connectionDestroy();
					return 0;
				}
				if (!connectWithTimeout(_sockfd, (struct sockaddr *)&serv_addr6, sizeof(serv_addr6), _connectTimeoutMs))
				{
					if (!_connectFailedReported)
					{
						std::cerr << "ERROR: Connection Failed";
						if (errno)
							std::cerr << ": " << std::strerror(errno);
						std::cerr << "\n";
						_connectFailedReported = true;
					}
					connectionDestroy();
					return 0;
				}
			}
		}
		else
		{
			serv_unaddr.sun_family = AF_UNIX;
			strcpy(serv_unaddr.sun_path, host);

			if (!connectWithTimeout(_sockfd, (struct sockaddr *)&serv_unaddr, SUN_LEN(&serv_unaddr), _connectTimeoutMs))
			{
				if (!_connectFailedReported)
				{
					std::cerr << "ERROR: Connection Failed";
					if (errno)
						std::cerr << ": " << std::strerror(errno);
					std::cerr << "\n";
					_connectFailedReported = true;
				}
				connectionDestroy();
				return 0;
			}
		}

		_connected = true;
		_connectFailedReported = false;

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
	ssize_t n = readFully(_sockfd, headbuff, sizeof(headbuff));
	if (n == 0)
	{
		connectionDestroy();
		std::cerr << "ERROR: Socket closed by peer" << std::endl;
		return -1;
	}

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

	n = readFully(_sockfd, recvBuff, messageSize);
	if (n == 0)
	{
		connectionDestroy();
		std::cerr << "ERROR: Socket closed by peer" << std::endl;
		return -1;
	}
	if (n < 0)
	{
		connectionDestroy();
		std::cerr << "ERROR: Read from socket failed" << std::endl;
		return -1;
	}
	parseReply(hyperhdrnet::GetReply(recvBuff));
	return 0;
}

void FlatBufferConnection::drainReplies()
{
	if (!_sockfd)
		return;

	uint8_t scratch[1024];
	for (;;)
	{
		ssize_t n = ::recv(_sockfd, scratch, sizeof(scratch), MSG_DONTWAIT);
		if (n > 0)
			continue;
		if (n == 0)
		{
			connectionDestroy();
			std::cerr << "ERROR: Socket closed by peer" << std::endl;
			return;
		}
		if (errno == EINTR)
			continue;
		return; // EAGAIN/EWOULDBLOCK: nothing pending
	}
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

	// Reduce FlatBufferBuilder realloc/memmove churn by pre-sizing once per
	// (growing) frame size. This does NOT remove the required copy into the
	// FlatBuffer, it just avoids repeated buffer growth.
	const size_t imgBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
	reserveBuilderBytes(imgBytes + 1024u);

	auto imgData = _builder.CreateVector(buffer, width * height * 3);
	auto rawImg = hyperhdrnet::CreateRawImage(_builder, imgData, width, height);
	auto imageReq = hyperhdrnet::CreateImage(_builder, hyperhdrnet::ImageType_RawImage, rawImg.Union(), -1);
	auto req = hyperhdrnet::CreateRequest(_builder, hyperhdrnet::Command_Image, imageReq.Union());
	_builder.Finish(req);

	int ret = sendMessage(_builder.GetBufferPointer(), _builder.GetSize());
	_builder.Clear();

	if (ret != -1)
	{
		if (_skipReply)
			drainReplies();
		else
			readData();
	}

	return ret;
}

int FlatBufferConnection::setNV12Image(const uint8_t *yPlane, const uint8_t *uvPlane, uint16_t width, uint16_t height, uint16_t strideY, uint16_t strideUV)
{
	if (!_registered)
	{
		if (setRegister(_origin, _priority) == -1)
			return -1;
	}

	// Send the full plane buffers (including per-row padding) and provide strides.
	const size_t yBytes = static_cast<size_t>(strideY) * static_cast<size_t>(height);
	const size_t uvBytes = static_cast<size_t>(strideUV) * (static_cast<size_t>(height) / 2u);
	reserveBuilderBytes(yBytes + uvBytes + 1024u);

	auto yData = _builder.CreateVector(yPlane, yBytes);
	auto uvData = _builder.CreateVector(uvPlane, uvBytes);
	auto nv12Img = hyperhdrnet::CreateNV12Image(_builder, yData, uvData, width, height, strideY, strideUV);
	auto imageReq = hyperhdrnet::CreateImage(_builder, hyperhdrnet::ImageType_NV12Image, nv12Img.Union(), -1);
	auto req = hyperhdrnet::CreateRequest(_builder, hyperhdrnet::Command_Image, imageReq.Union());
	_builder.Finish(req);

	int ret = sendMessage(_builder.GetBufferPointer(), _builder.GetSize());
	_builder.Clear();

	if (ret != -1)
	{
		if (_skipReply)
			drainReplies();
		else
			readData();
	}

	return ret;
}

int FlatBufferConnection::sendPrebuiltRequest(const uint8_t *buffer, uint32_t size)
{
	if (!_registered)
	{
		if (setRegister(_origin, _priority) == -1)
			return -1;
	}

	int ret = sendMessage(buffer, size);
	if (ret != -1)
	{
		if (_skipReply)
			drainReplies();
		else
			readData();
	}

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

	struct iovec iov[2];
	iov[0].iov_base = const_cast<uint8_t *>(header);
	iov[0].iov_len = sizeof(header);
	iov[1].iov_base = const_cast<uint8_t *>(buffer);
	iov[1].iov_len = size;

	if (writevFully(_sockfd, iov, 2) < 0)
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
			std::cout << "Set video mode " << videoMode << "\n";
			return true;
		}

		// We got a registered reply.
		if (registered == _priority)
		{
			if (!_registered)
				std::cout << "Client registered succesfully\n";
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
