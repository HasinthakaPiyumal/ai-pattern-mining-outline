# Cluster 30

def get_changelog(github_timestamp):
    release_date = datetime.fromisoformat(github_timestamp[:-1])
    utc_timezone = pytz.timezone('UTC')
    release_date = utc_timezone.localize(release_date)
    os.chdir(EvaDB_DIR)
    run_command('git pull origin master')
    repo = git.Repo('.git')
    regexp = re.compile('\\#[0-9]*')
    changelog = ''
    for commit in repo.iter_commits('master'):
        if commit.authored_datetime < release_date:
            break
        output = regexp.search(commit.message)
        if '[BUMP]' in commit.message:
            continue
        if output is None:
            continue
        else:
            pr_number = output.group(0)
            key_message = commit.message.split('\n')[0]
            key_message = key_message.split('(')[0]
            pr_number = pr_number.split('#')[1]
            if '[RELEASE]' in key_message:
                continue
            changelog += f'* PR #{pr_number}: {key_message}\n'
    return changelog

def run_command(command_str: str):
    output = subprocess.check_output(command_str, shell=True, universal_newlines=True).rstrip()
    pprint(command_str)
    if 'version' in command_str:
        pprint(output)
    return output

def release_version(current_version):
    version_path = os.path.join(os.path.join(EvaDB_DIR, 'evadb'), 'version.py')
    with open(version_path, 'r') as version_file:
        output = version_file.read()
        output = output.replace('+dev', '')
    with open(version_path, 'w') as version_file:
        version_file.write(output)
    NEXT_RELEASE = current_version
    run_command('git checkout -b release-' + NEXT_RELEASE)
    run_command('git add . -u')
    run_command("git commit -m '[RELEASE]: " + NEXT_RELEASE + "'")
    run_command('git push --set-upstream origin release-' + NEXT_RELEASE)
    run_command(f'git push origin release-{NEXT_RELEASE}')

def publish_wheels(tag):
    run_command('rm -rf dist build')
    run_command('python3 setup.py sdist')
    run_command('python3 setup.py bdist_wheel')
    run_command(f'python3 -m pip install dist/evadb-{tag}-py3-none-any.whl')
    run_command('python3 -c "import evadb; print(evadb.__version__)" ')
    print('Running twine to upload wheels')
    print('Ensure that you have .pypirc file in your $HOME folder')
    run_command('twine upload dist/* -r pypi')

def bump_up_version(next_version):
    version_path = os.path.join(os.path.join(EvaDB_DIR, 'evadb'), 'version.py')
    major_str = get_string_in_line(version_path, 1)
    minor_str = get_string_in_line(version_path, 2)
    patch_str = get_string_in_line(version_path, 3)
    assert 'dev' not in patch_str
    major_str = f'_MAJOR = "{str(next_version.major)}"\n'
    minor_str = f'_MINOR = "{str(next_version.minor)}"\n'
    patch_str = f'_REVISION = "{str(next_version.patch)}+dev"\n\n'
    footer = 'VERSION_SHORT = f"{_MAJOR}.{_MINOR}"\nVERSION = f"{_MAJOR}.{_MINOR}.{_REVISION}"'
    output = major_str + minor_str + patch_str + footer
    with open(version_path, 'w') as version_file:
        version_file.write(output)
    NEXT_RELEASE = f'v{str(next_version)}+dev'
    run_command('git checkout -b bump-' + NEXT_RELEASE)
    run_command('git add . -u')
    run_command("git commit -m '[BUMP]: " + NEXT_RELEASE + "'")
    run_command('git push --set-upstream origin bump-' + NEXT_RELEASE)
    run_command(f"gh pr create -B staging -H bump-{NEXT_RELEASE} --title 'Bump Version to {NEXT_RELEASE}' --body 'Bump Version to {NEXT_RELEASE}'")

def get_string_in_line(file_path, line_number):
    line = linecache.getline(file_path, line_number)
    return line.strip()

def get_commit_id_of_latest_release():
    import requests
    repo = 'georgia-tech-db/evadb'
    url = f'https://api.github.com/repos/{repo}/releases'
    response = requests.get(url)
    data = response.json()
    latest_release = data[0]
    release_date = latest_release['created_at']
    return release_date

def append_changelog(insert_changelog: str, version: str):
    with open(EvaDB_CHANGELOG_PATH, 'r') as file:
        file_contents = file.read()
    position = 327
    version = version.split('+')[0]
    today_date = date.today()
    header = f'##  [{version}] - {today_date}\n\n'
    modified_content = file_contents[:position] + header + insert_changelog + '\n' + file_contents[position:]
    with open(EvaDB_CHANGELOG_PATH, 'w') as file:
        file.write(modified_content)

def upload_assets(changelog, tag):
    access_token = os.environ['GITHUB_KEY']
    repo_owner = 'georgia-tech-db'
    repo_name = 'evadb'
    tag_name = 'v' + tag
    assert 'vv' not in tag
    asset_filepaths = [f'dist/evadb-{tag}-py3-none-any.whl', f'dist/evadb-{tag}.tar.gz']
    g = Github(access_token)
    repo = g.get_repo(f'{repo_owner}/{repo_name}')
    release_name = tag_name
    release_message = f'{tag_name}\n\n {changelog}'
    release = repo.create_git_release(tag=tag_name, name=release_name, message=release_message, draft=False, prerelease=False)
    release = repo.get_release(tag_name)
    for filepath in asset_filepaths:
        asset_name = filepath.split('/')[-1]
        asset_path = Path(f'{EvaDB_DIR}/{filepath}')
        print('path: ' + str(asset_path))
        release.upload_asset(str(asset_path), asset_name)
    print('Release created and published successfully.')

