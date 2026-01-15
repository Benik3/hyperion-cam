#include "event_loop.h"

#include <assert.h>
#include <event2/event.h>
#include <event2/thread.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <mutex>

EventLoop::EventLoop()
{
	assert(!instance_);

	static std::once_flag once;
	std::call_once(once, []()
				   { evthread_use_pthreads(); });

	event_ = event_base_new();
	assert(event_);

	wakeupFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
	assert(wakeupFd_ >= 0);

	wakeupEvent_ = event_new(event_, wakeupFd_, EV_READ | EV_PERSIST, &EventLoop::wakeupCb, this);
	assert(wakeupEvent_);
	event_add(wakeupEvent_, nullptr);

	instance_ = this;
}

EventLoop::~EventLoop()
{
	instance_ = nullptr;

	if (wakeupEvent_)
		event_free(wakeupEvent_);
	wakeupEvent_ = nullptr;

	if (wakeupFd_ >= 0)
		::close(wakeupFd_);
	wakeupFd_ = -1;

	event_base_free(event_);
	libevent_global_shutdown();
}

int EventLoop::exec()
{
	exitCode_ = -1;
	exit_.store(false, std::memory_order_release);

	// All wakeups and cross-thread callLater() use wakeupFd_.
	// event_base_dispatch() blocks efficiently until we are woken.
	while (!exit_.load(std::memory_order_acquire))
	{
		dispatchCalls();
		if (exit_.load(std::memory_order_acquire))
			break;
		event_base_dispatch(event_);
	}

	return exitCode_;
}

void EventLoop::exit(int code)
{
	exitCode_ = code;
	exit_.store(true, std::memory_order_release);
	interrupt();
}

void EventLoop::interrupt()
{
	// Wake the loop even if it's blocked in poll().
	uint64_t one = 1;
	(void)::write(wakeupFd_, &one, sizeof(one));
	event_base_loopbreak(event_);
}

void EventLoop::enqueue(Task &&task)
{
	bool needWake = false;
	{
		std::unique_lock<std::mutex> locker(lock_);
		calls_.push_back(std::move(task));
		// Only wake the event loop when we transition from "no pending wake" to "need wake".
		needWake = !wakeupPending_.exchange(true, std::memory_order_acq_rel);
	}

	if (needWake)
	{
		uint64_t one = 1;
		(void)::write(wakeupFd_, &one, sizeof(one));
	}
}

void EventLoop::dispatchCalls()
{
	std::deque<Task> local;
	{
		std::unique_lock<std::mutex> locker(lock_);
		if (calls_.empty())
		{
			wakeupPending_.store(false, std::memory_order_release);
			return;
		}
		local.swap(calls_);
		wakeupPending_.store(false, std::memory_order_release);
	}

	for (auto &call : local)
		call();
}

void EventLoop::wakeupCb(evutil_socket_t, short, void *arg)
{
	auto *self = static_cast<EventLoop *>(arg);
	// Drain eventfd (coalesce multiple wakeups).
	uint64_t val;
	for (;;)
	{
		ssize_t n = ::read(self->wakeupFd_, &val, sizeof(val));
		if (n == static_cast<ssize_t>(sizeof(val)))
			continue;
		if (n < 0 && (errno == EAGAIN || errno == EINTR))
			break;
		break;
	}
	self->dispatchCalls();
}
