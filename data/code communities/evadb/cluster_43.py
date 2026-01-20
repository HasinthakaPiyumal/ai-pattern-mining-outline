# Cluster 43

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file detected')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path = Path(UPLOAD_FOLDER + '/' + filename)
            if path.is_file():
                flash('File uploaded')
                return {'response': 'File saved'}
            else:
                flash('Failed to upload file')
                return redirect(request.url)
    return '<html>\n                <body>\n                    <form method="POST"\n                        enctype = "multipart/form-data">\n                        <input type = "file" name = "file" />\n                        <input type = "submit"/>\n                    </form>\n                </body>\n            </html>'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

