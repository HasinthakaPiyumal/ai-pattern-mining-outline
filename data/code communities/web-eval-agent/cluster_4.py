# Cluster 4

def get_browser_manager() -> PlaywrightBrowserManager:
    """Get the singleton browser manager instance.
    
    This function provides a centralized way to access the singleton
    PlaywrightBrowserManager instance throughout the application.
    
    Returns:
        PlaywrightBrowserManager: The singleton browser manager instance
    """
    return PlaywrightBrowserManager.get_instance()

def get_web_evaluation_prompt(url: str, task: str) -> str:
    """
    Generate a prompt for web application evaluation.
    
    Args:
        url: The URL of the web application to evaluate
        task: The specific aspect to test
        
    Returns:
        str: The formatted evaluation prompt
    """
    return f"""VISIT: {url}\nGOAL: {task}\n\nEvaluate the UI/UX of the site. If you hit any critical errors (e.g., page fails to load, JS errors), stop and report the exact issue.\n\nIf a login page appears, first try clicking "Login" — saved credentials may work.\nIf login fields appear and no credentials are provided, do not guess. Stop and report that login is required. Suggest the user run setup_browser_state to log in and retry.\n\nIf no errors block progress, proceed and attempt the task. Try a couple times if needed before giving up — unless blocked by missing login access.\nMake sure to click through the application from the base url, don't jump to other pages without naturally arriving there.\n\nReport any UX issues (e.g., incorrect content, broken flows), or confirm everything worked smoothly.\nTake note of any opportunities for improvement in the UI/UX, test and think about the application like a real user would.\n"""

