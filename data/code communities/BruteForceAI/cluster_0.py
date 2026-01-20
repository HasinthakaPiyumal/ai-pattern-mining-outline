# Cluster 0

def execute_check_updates(args):
    """Execute Check Updates"""
    print('🔄 BruteForceAI Update Check')
    print('=' * 50)
    result = check_for_updates(silent=False, force=True)
    if result is None:
        print('❌ Update check failed')
    elif result.get('update_available'):
        print('🎉 Update check completed - update available!')
    else:
        print("✅ Update check completed - you're up to date!")

def check_for_updates(silent=False, force=False):
    """
    Check for updates from mordavid.com
    
    Args:
        silent: If True, only show update messages, not "up to date" messages
        force: If True, force check even if checked recently
    
    Returns:
        dict: Update information or None if check failed
    """
    try:
        response = requests.get(VERSION_CHECK_URL, timeout=3)
        response.raise_for_status()
        data = yaml.safe_load(response.text)
        bruteforce_info = None
        for software in data.get('softwares', []):
            if software.get('name', '').lower() == 'bruteforceai':
                bruteforce_info = software
                break
        if not bruteforce_info:
            return None
        latest_version = bruteforce_info.get('version', '0.0.0')
        if latest_version != CURRENT_VERSION:
            print(f'🔄 Update available: v{CURRENT_VERSION} → v{latest_version} | Download: {bruteforce_info.get('url', 'N/A')}\n')
            return {'update_available': True, 'current_version': CURRENT_VERSION, 'latest_version': latest_version, 'info': bruteforce_info}
        else:
            if not silent:
                print(f'✅ BruteForceAI v{CURRENT_VERSION} is up to date\n')
            return {'update_available': False, 'current_version': CURRENT_VERSION, 'latest_version': latest_version}
    except:
        return None

