# Cluster 21

@app.on_event('startup')
def on_startup():
    start_scheduler()

def start_scheduler():
    if not scheduler.running:
        logger.info('start scheduler of sync_hxd_task_respone')
        scheduler.add_job(sync_hxd_task_response, IntervalTrigger(seconds=1))
        scheduler.add_job(fetch_chat_response, IntervalTrigger(seconds=1))
        scheduler.start()

