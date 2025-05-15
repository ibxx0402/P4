// Server side implementation of UDP client-server model 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <algorithm>
#include <ctime>
#include <vector>
#include <queue>

//since it is written in c instead of c++
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

#define SERVER_IP "192.168.0.125"
#define CLIENT_PORT	 9998
#define CAMERA_PORT 9999 
#define MAXLINE 1024 
#define MAX_UDP_SIZE 65507  

struct FFmpegContext {
    const AVCodec* codec;  
    AVCodecContext* context;
    AVFrame* frame_yuv;
};

FFmpegContext m_ffmpeg;

std::vector<uint8_t> h264_buffer; // Buffer to accumulate H.264 data

bool isCompleteNalUnit(const std::vector<uint8_t>& buffer) {
    // Check if the buffer has at least one NAL unit
    // Look for start code pattern: 0x00 0x00 0x01 or 0x00 0x00 0x00 0x01
    size_t i = 0;
    int start_codes_found = 0;

    while (i < buffer.size() - 3) {
        if ((buffer[i] == 0 && buffer[i+1] == 0 && buffer[i+2] == 1) ||
            (i < buffer.size() - 4 && buffer[i] == 0 && buffer[i+1] == 0 && 
             buffer[i+2] == 0 && buffer[i+3] == 1)) {
            
            start_codes_found++;
            if (start_codes_found >= 2) {
                return true; // Found at least 2 start codes (one complete NAL unit)
            }
            
            // Skip ahead to avoid finding the same start code
            i += 3;
        } else {
            i++;
        }
    }
    
    return false; // Not enough start codes found
}

// Function to find start of NAL unit in buffer
size_t findNalStartCode(const std::vector<uint8_t>& buffer, size_t start_pos = 0) {
    for (size_t i = start_pos; i < buffer.size() - 3; i++) {
        // Check for 3-byte start code: 0x00 0x00 0x01
        if (buffer[i] == 0 && buffer[i+1] == 0 && buffer[i+2] == 1) {
            return i;
        }
        // Check for 4-byte start code: 0x00 0x00 0x00 0x01
        if (i < buffer.size() - 4 && buffer[i] == 0 && buffer[i+1] == 0 && 
            buffer[i+2] == 0 && buffer[i+3] == 1) {
            return i;
        }
    }
    return buffer.size(); // Not found
}

// Driver code 
int main() { 
	// Initialize libav used to decode H.264
	avformat_network_init();
	m_ffmpeg.codec = avcodec_find_decoder(AV_CODEC_ID_H264);
	if (!m_ffmpeg.codec) 
	{
		fprintf(stderr, "Codec not found\n");
		exit(1);
	}
	m_ffmpeg.context = avcodec_alloc_context3(m_ffmpeg.codec);
	if (!m_ffmpeg.context)
	{
		fprintf(stderr, "Could not allocate video codec context\n");
		exit(1);
	}

	// Add error resilience flags
	m_ffmpeg.context->err_recognition = AV_EF_CAREFUL;
	m_ffmpeg.context->flags |= AV_CODEC_FLAG_LOW_DELAY;
	m_ffmpeg.context->flags2 |= AV_CODEC_FLAG2_CHUNKS;
	m_ffmpeg.context->thread_count = 4;
 	m_ffmpeg.context->thread_type = FF_THREAD_SLICE;
	m_ffmpeg.context->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

	// Add extradata for SPS/PPS (you may need to adjust this based on your stream)
	// If you know the SPS/PPS of your Raspberry Pi stream, you can set it here

	if (avcodec_open2(m_ffmpeg.context, m_ffmpeg.codec, nullptr) < 0)
	{
		fprintf(stderr, "Could not open codec\n");
		exit(1);
	}

	// Use av_frame_alloc instead of avcodec_alloc_frame
	m_ffmpeg.frame_yuv = av_frame_alloc();
	if (!m_ffmpeg.frame_yuv) 
	{
		fprintf(stderr, "Could not allocate video frame\n");
		exit(1);
	}

	if (!m_ffmpeg.frame_yuv) {
		fprintf(stderr, "Decoded frame is null\n");
		exit(1);
	}

	char buffer[MAXLINE]; 

	//Creating client socket
	//###################################################################################
	int client_sock; 
	struct sockaddr_in client_bind_addr, client_peer_addr; 
	
	// Creating socket file descriptor 
	if ( (client_sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
		perror("socket creation failed"); 
		exit(EXIT_FAILURE); 
	} 
	
	memset(&client_bind_addr, 0, sizeof(client_bind_addr)); 
	memset(&client_peer_addr, 0, sizeof(client_peer_addr)); 
	
	// Filling server information for client
	client_bind_addr.sin_family = AF_INET; // IPv4 
	client_bind_addr.sin_addr.s_addr =  inet_addr(SERVER_IP); 
	client_bind_addr.sin_port = htons(CLIENT_PORT); 
	
	// Bind the socket with the server address 
	if ( bind(client_sock, (const struct sockaddr *)&client_bind_addr, 
			sizeof(client_bind_addr)) < 0 ) 
	{ 
		perror("bind failed"); 
		exit(EXIT_FAILURE); 
	} 
	
	//##################################################################################

	//Creating server socket
	//##################################################################################
	int camera_sock; 
	struct sockaddr_in camera_addr, camera_peer_addr; 
	
	// Creating socket file descriptor 
	if ( (camera_sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
		perror("socket creation failed"); 
		exit(EXIT_FAILURE); 
	} 
	
	memset(&camera_addr, 0, sizeof(camera_addr)); 
	memset(&camera_peer_addr, 0, sizeof(camera_peer_addr)); 
	
	// Filling server information for client
	camera_addr.sin_family = AF_INET; // IPv4 
	camera_addr.sin_addr.s_addr =  inet_addr(SERVER_IP); 
	camera_addr.sin_port = htons(CAMERA_PORT); 

	// After creating camera_sock, add:
	int enable = 1;
	if (setsockopt(camera_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
		perror("setsockopt(SO_REUSEADDR) failed");
	}

	// Set receive buffer size to be large enough for video frames
	int rcvbuf = 1024 * 1024; // 1MB buffer
	if (setsockopt(camera_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0) {
		perror("setsockopt(SO_RCVBUF) failed");
	}
	
	// Bind the socket with the server address 
	if ( bind(camera_sock, (const struct sockaddr *)&camera_addr, 
			sizeof(camera_addr)) < 0 ) 
	{ 
		perror("bind failed"); 
		exit(EXIT_FAILURE); 
	} 
	//##################################################################################
	std::cout<<"Server: Listening for client registration on "<< SERVER_IP << ":" << CLIENT_PORT << " & " << CAMERA_PORT <<std::endl; 
	
	// Add this above your main loop:
	struct sockaddr_in registered_client_addr;
	socklen_t registered_client_len = 0;
	bool client_registered = false;

	while (true) {
		fd_set readfds;
		FD_ZERO(&readfds);
		FD_SET(client_sock, &readfds);
		FD_SET(camera_sock, &readfds);

		int maxfd = std::max(client_sock, camera_sock);

		int activity = select(maxfd + 1, &readfds, NULL, NULL, NULL); // Block forever

		if (activity > 0) {
			if (FD_ISSET(client_sock, &readfds)) {
				// Handle client_sock activity
				char buffer[1024];
				struct sockaddr_in from_addr;
				socklen_t from_len = sizeof(from_addr);

				ssize_t n = recvfrom(client_sock, buffer, sizeof(buffer), 0,
									 (struct sockaddr *)&from_addr, &from_len);

				if (n > 0) {
					buffer[n] = '\0'; // Null-terminate if you expect string data
					std::cout << "Server: Received client message from "
							  << inet_ntoa(from_addr.sin_addr) << ":" << ntohs(from_addr.sin_port)
							  << " : " << buffer << std::endl;

					 // Store the client address for forwarding video
					registered_client_addr = from_addr;
					registered_client_len = from_len;
					client_registered = true;

					// After printing the client message
					const char* confirmation = "Registration successful";
					sendto(client_sock, confirmation, strlen(confirmation), 0, (struct sockaddr *)&from_addr, from_len);
				}
				
			}
			if (FD_ISSET(camera_sock, &readfds)) {
				static unsigned long long total_bytes = 0;
				static int packets = 0;
				static time_t last_stats = time(nullptr);
				
				char buffer[65536];
				struct sockaddr_in from_addr; 
				socklen_t from_len = sizeof(from_addr);
				
				int data = recvfrom(camera_sock, buffer, sizeof(buffer), 0, 
									(struct sockaddr *)&from_addr, &from_len);
									
				if (data > 0 && client_registered) {
					// Extend our H.264 buffer with new data
					h264_buffer.insert(h264_buffer.end(), buffer, buffer + data);
					
					 // Process complete NAL units from the buffer
					while (isCompleteNalUnit(h264_buffer)) {
						size_t nal_start = findNalStartCode(h264_buffer);
						size_t next_nal_start = findNalStartCode(h264_buffer, nal_start + 3);
						
						if (next_nal_start > nal_start && next_nal_start < h264_buffer.size()) {
							// Extract complete NAL unit
							std::vector<uint8_t> nal_unit(h264_buffer.begin() + nal_start, h264_buffer.begin() + next_nal_start);
							
							// Create packet properly
							AVPacket *packet = av_packet_alloc();
							if (!packet) {
								std::cerr << "Failed to allocate packet" << std::endl;
								continue;
							}
							
							// Allocate and copy data properly
							uint8_t* packet_data = (uint8_t*)av_malloc(nal_unit.size());
							if (!packet_data) {
								std::cerr << "Failed to allocate packet data buffer" << std::endl;
								av_packet_free(&packet);
								continue;
							}
							
							// Copy NAL unit to packet data buffer
							memcpy(packet_data, nal_unit.data(), nal_unit.size());
							
							// Set up packet with proper buffer management
							av_packet_from_data(packet, packet_data, nal_unit.size());
							packet->pts = AV_NOPTS_VALUE;
							packet->dts = AV_NOPTS_VALUE;

							// Send the packet to the decoder
							int send_result = avcodec_send_packet(m_ffmpeg.context, packet);

							if (send_result == 0) {
								// Try to receive decoded frame
								int receive_result = avcodec_receive_frame(m_ffmpeg.context, m_ffmpeg.frame_yuv);
								
								if (receive_result == 0) {
									//YUV to BGR conversion using all planes
									int width = m_ffmpeg.frame_yuv->width;
									int height = m_ffmpeg.frame_yuv->height;

									// Create a YUV image with the correct size
									cv::Mat yuv_img(height + height/2, width, CV_8UC1);
									
									// Copy Y plane (full resolution)
									memcpy(yuv_img.data, m_ffmpeg.frame_yuv->data[0], 
										   width * height);
									
									// Copy U and V planes (quarter resolution each)
									memcpy(yuv_img.data + width * height, 
										   m_ffmpeg.frame_yuv->data[1], 
										   width * height / 4);
									memcpy(yuv_img.data + width * height + width * height / 4, 
										   m_ffmpeg.frame_yuv->data[2], 
										   width * height / 4);
									
									// Now convert the properly formatted YUV data to BGR
									cv::Mat bgr_frame;
									cv::cvtColor(yuv_img, bgr_frame, cv::COLOR_YUV420p2BGR);
									
									// Display the frame
									cv::imshow("Video", bgr_frame);
									
									if (cv::waitKey(1) == 27) { // Exit on 'ESC' key
										break;
									} 
								} else {
									std::cerr << "Error decoding frame: " << av_err2str(receive_result) << std::endl;
									if (receive_result == AVERROR(EAGAIN)) {
										std::cout << "Decoder needs more data" << std::endl;
									} else if (receive_result == AVERROR_EOF) {
										std::cout << "End of stream reached" << std::endl;
									} else {
										// Reset the decoder after serious errors
										avcodec_flush_buffers(m_ffmpeg.context);
									}
								}
								
								// Unref the frame to prepare for next decode
								av_frame_unref(m_ffmpeg.frame_yuv);
							}

							// Free the packet
							av_packet_free(&packet);
							
							// Remove processed NAL unit from buffer
							h264_buffer.erase(h264_buffer.begin(), h264_buffer.begin() + next_nal_start);
						} else {
							// Not enough data for a complete NAL unit
							break;
						}
					}
					
					// If buffer gets too large, trim it
					if (h264_buffer.size() > 1000000) { // 1MB threshold
						size_t nal_start = findNalStartCode(h264_buffer);
						if (nal_start < h264_buffer.size()) {
							h264_buffer.erase(h264_buffer.begin(), h264_buffer.begin() + nal_start);
						} else {
							// No start code found, discard half the buffer
							h264_buffer.erase(h264_buffer.begin(), h264_buffer.begin() + h264_buffer.size()/2);
						}
						std::cout << "Buffer trimmed to " << h264_buffer.size() << " bytes" << std::endl;
					}
				}
			}
		}
	}

	// Free FFmpeg resources
	if (m_ffmpeg.frame_yuv) {
		av_frame_free(&m_ffmpeg.frame_yuv);
	}
	if (m_ffmpeg.context) {
		avcodec_free_context(&m_ffmpeg.context);
	}

	// Close sockets
	close(client_sock);
	close(camera_sock);

	return 0; 
}