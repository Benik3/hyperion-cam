#ifndef __FLATBUFFER_CONNECTION_H__
#define __FLATBUFFER_CONNECTION_H__

#include <flatbuffers/flatbuffers.h>

namespace hyperhdrnet
{
	struct Reply;
}

///
/// Connection class to setup an connection to the hyperion server and execute commands.
///
class FlatBufferConnection
{

public:
	/// @brief Constructor
	FlatBufferConnection();

	/// @brief Destructor
	~FlatBufferConnection();

	/// @brief Connect to TCP socket
	/// @param origin
	/// @param host
	/// @param port
	/// @param priority
	/// @param skipReply
	/// @return  1 = connected, 0 = disconnected
	int connectToServer(const char *origin, const char *host, uint16_t port, int priority, bool skipReply);

	/// @brief Destroy connection and reset flags
	void connectionDestroy();

	/// @brief Register flatbuffer grabber in hyperion
	/// @param origin
	/// @param priority
	/// @return 0 = OK, -1 = failed
	int setRegister(const char *origin, int priority);

	/// @brief Send image data through flatbuffer
	/// @param buffer
	/// @param width
	/// @param height
	/// @return 0 = OK, -1 = failed
	int setImage(const uint8_t *buffer, uint16_t width, uint16_t height);

	/// @brief Send NV12 image data through flatbuffer
	/// @param yPlane Pointer to Y plane
	/// @param uvPlane Pointer to interleaved UV plane
	/// @param width Image width (pixels)
	/// @param height Image height (pixels)
	/// @param strideY Y plane stride (bytes per row)
	/// @param strideUV UV plane stride (bytes per row)
	/// @return 0 = OK, -1 = failed
	int setNV12Image(const uint8_t *yPlane, const uint8_t *uvPlane, uint16_t width, uint16_t height, uint16_t strideY, uint16_t strideUV);

	/// @brief Pre-size the internal FlatBufferBuilder to reduce reallocations.
	/// Safe to call multiple times; it only grows the internal buffer.
	void reserveBuilderBytes(size_t bytes);

	/// @brief Set connect timeout used for establishing the socket connection.
	/// Applies to TCP/Unix socket connect attempts.
	void setConnectTimeoutMs(int timeoutMs);

	/// @brief Set send timeout (SO_SNDTIMEO) for the data socket.
	/// Bounds how long a blocking write can stall if the server stops reading.
	void setSendTimeoutMs(int timeoutMs);

	/// @brief Send a prebuilt FlatBuffers Request message.
	///
	/// This avoids copying image payloads into FlatBufferConnection's internal
	/// FlatBufferBuilder. The buffer must contain a complete HyperHDR Request
	/// flatbuffer (without the 4-byte length prefix).
	///
	/// @return 0 = OK, -1 = failed
	int sendPrebuiltRequest(const uint8_t *buffer, uint32_t size);

private:
	/// @brief Send data through socket
	/// @param buffer
	/// @param size
	/// @return 0 = OK, -1 = failed
	int sendMessage(const uint8_t *buffer, uint32_t size);

	/// @brief Read data from socket
	/// @return 0 = OK, -1 = failed
	int readData();

	///
	/// @brief Parse a reply message
	/// @param reply The received reply
	/// @return true if the reply indicates success
	///
	bool parseReply(const hyperhdrnet::Reply *reply);

private:
	/// The TCP-Socket with the connection to the server
	int _sockfd;

	const char *_origin;
	int _priority;

	/// Host address
	const char *_host;

	/// Host port
	uint16_t _port;

	/// buffer for reply
	unsigned char recvBuff[1024];

	flatbuffers::FlatBufferBuilder _builder;

	size_t _reservedBuilderBytes{0};

	bool _registered;

	bool _connected;

	bool _skipReply;

	// Log gating: print connection failure once until a successful connect clears it.
	bool _connectFailedReported{false};

	int _connectTimeoutMs{250};

	int _sendTimeoutMs{5000};
};

#endif