# Cluster 22

@app.on_event('shutdown')
def on_shutdown():
    stop_scheduler()

def stop_scheduler():
    task_1 = 'sync_hxd_task_response'
    if release_scheduler_lock(task_1):
        logger.info(f'release scheduler lock of {task_1} successfully.')
    else:
        logger.error(f'release scheduler lock of {task_1} failed. you should delete this key from redis manually.')

def release_scheduler_lock(task) -> bool:
    key = f'{biz_const.RDS_KEY_SCHEDULER}-{task}'
    r.delete(key)
    return False if r.exists(key) else True

