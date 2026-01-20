# Cluster 5

def display_message_with_sheets_upload(message, message_index):
    content = message['content']
    if isinstance(content, (str, bytes, BytesIO)):
        data = extract_data_from_markdown(content)
        if data is not None:
            try:
                is_excel = isinstance(data, BytesIO) or (isinstance(content, str) and 'excel' in content.lower())
                if is_excel:
                    df = format_data(data, 'excel')
                else:
                    df = format_data(data, 'csv')
                if df is not None:
                    st.dataframe(df)
                    if not is_excel:
                        csv_buffer = BytesIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        st.download_button(label='📥 Download as CSV', data=csv_buffer, file_name='data.csv', mime='text/csv', key=f'csv_download_{message_index}')
                    else:
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name='Sheet1')
                        excel_buffer.seek(0)
                        st.download_button(label='📥 Download as Excel', data=excel_buffer, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key=f'excel_download_{message_index}')
                    display_google_sheets_button(df, f'sheets_upload_{message_index}')
                else:
                    st.warning('Failed to display data as a table. Showing raw content:')
                    st.code(content)
            except Exception as e:
                st.error(f'Error processing data: {str(e)}')
                st.code(content)
        else:
            st.markdown(content)
    else:
        st.markdown(str(content))

def extract_data_from_markdown(text: Union[str, bytes, io.BytesIO]) -> Union[str, bytes, io.BytesIO, None]:
    if isinstance(text, io.BytesIO):
        return text
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    pattern = '```(csv|excel)\\n(.*?)\\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        data_type = match.group(1)
        data = match.group(2).strip()
        if data_type == 'excel':
            return io.BytesIO(data.encode())
        return data
    return None

def format_data(data: Union[str, bytes, io.BytesIO], format_type: str):
    try:
        if isinstance(data, io.BytesIO):
            if format_type == 'excel':
                return pd.read_excel(data, engine='openpyxl')
            data.seek(0)
            return pd.read_csv(data)
        elif isinstance(data, bytes):
            if format_type == 'excel':
                return pd.read_excel(io.BytesIO(data), engine='openpyxl')
            return pd.read_csv(io.BytesIO(data))
        elif format_type == 'csv':
            csv_data = []
            csv_reader = csv.reader(io.StringIO(data))
            for row in csv_reader:
                csv_data.append(row)
            if not csv_data:
                raise ValueError('Empty CSV data')
            max_columns = max((len(row) for row in csv_data))
            padded_data = [row + [''] * (max_columns - len(row)) for row in csv_data]
            headers = padded_data[0]
            unique_headers = []
            for i, header in enumerate(headers):
                if header == '' or header in unique_headers:
                    unique_headers.append(f'Column_{i + 1}')
                else:
                    unique_headers.append(header)
            df = pd.DataFrame(padded_data[1:], columns=unique_headers)
            df = df.loc[:, (df != '').any(axis=0)]
            return df
        elif format_type == 'excel':
            return pd.read_excel(io.BytesIO(data.encode()), engine='openpyxl')
    except Exception as e:
        st.error(f'Error formatting data: {str(e)}')
        st.error(f'Data type: {type(data)}')
        st.error(f'Data content: {(data[:100] if isinstance(data, (str, bytes)) else 'BytesIO object')}')
        st.text('Raw data (first 500 characters):')
        st.text(data[:500] if isinstance(data, (str, bytes)) else 'BytesIO object')
        return None

def display_message(message):
    content = message['content']
    if isinstance(content, (str, bytes, io.BytesIO)):
        data = extract_data_from_markdown(content)
        if data is not None:
            if isinstance(data, io.BytesIO) or (isinstance(content, str) and 'excel' in content.lower()):
                df = format_data(data, 'excel')
            else:
                df = format_data(data, 'csv')
            if df is not None:
                st.dataframe(df)
            else:
                st.warning('Failed to display data as a table. Showing raw content:')
                st.code(content)
        else:
            st.markdown(content)
    else:
        st.markdown(str(content))

