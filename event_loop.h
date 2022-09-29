#ifndef __SIMPLE_CAM_EVENT_LOOP_H__
#define __SIMPLE_CAM_EVENT_LOOP_H__

#include <atomic>
#include <functional>
#include <list>
#include <mutex>

struct event_base;

class EventLoop
{
public:
	EventLoop();
	~EventLoop();

	void exit(int code = 0);
	int exec();

	void callLater(const std::function<void()> &func);

private:
	EventLoop *instance_ = nullptr;

	struct event_base *event_;
	std::atomic<bool> exit_;
	int exitCode_;

	std::list<std::function<void()>> calls_;
	std::mutex lock_;

	void interrupt();
	void dispatchCalls();
};

#endif /* __SIMPLE_CAM_EVENT_LOOP_H__ */
