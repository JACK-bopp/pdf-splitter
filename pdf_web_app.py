#!/usr/bin/env python3
"""PDF分割工具 """
import os
import sys
import json
import threading
import webbrowser
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import zipfile
import tempfile
import shutil

# 添加项目路径
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from pdf_processor import PDFProcessor, LargeFilePDFProcessor, comprehensive_integrity_check, create_integrity_report
from pdf_chapter_splitter import PDFChapterSplitter, validate_pdf_integrity
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pdf_splitter_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'pdf_uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(tempfile.gettempdir(), 'pdf_outputs')

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 全局变量存储处理状态
processing_status = {
    'is_processing': False,
    'progress': 0,
    'message': '准备就绪',
    'files': [],
    'error': None
}

def progress_callback(current, total, status_info=None):
    """进度回调函数"""
    global processing_status
    processing_status['progress'] = int((current / total) * 100)
    if status_info and 'extra_info' in status_info:
        processing_status['message'] = status_info['extra_info']

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """文件上传 - 支持单文件和多文件"""
    global processing_status
    
    if 'files' not in request.files:
        return jsonify({'error': '没有选择文件'})
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': '没有选择文件'})
    
    uploaded_files = []
    
    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # 获取文件信息
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                if file_size_mb > 100:
                    processor = LargeFilePDFProcessor()
                else:
                    processor = PDFProcessor()
                
                page_count, file_size_mb, detailed_info = processor.get_pdf_info(filepath)
                
                uploaded_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'page_count': page_count,
                    'file_size_mb': round(file_size_mb, 2),
                    'is_large_file': detailed_info.get('is_large_file', False),
                    'avg_page_size_kb': round(detailed_info.get('avg_page_size_kb', 0), 1)
                })
                
            except Exception as e:
                return jsonify({'error': f'文件分析失败: {str(e)}'})
    
    if not uploaded_files:
        return jsonify({'error': '没有有效的PDF文件'})
    
    return jsonify({
        'success': True,
        'files': uploaded_files,
        'count': len(uploaded_files)
    })

@app.route('/preview_pdf', methods=['POST'])
def preview_pdf():
    """预览PDF文件内容"""
    filepath = request.json.get('filepath')
    page_num = request.json.get('page', 1)  # 默认预览第一页
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'})
    
    try:
        import fitz  # PyMuPDF
        
        # 打开PDF文件
        doc = fitz.open(filepath)
        if page_num < 1 or page_num > len(doc):
            page_num = 1
        
        # 获取指定页面
        page = doc[page_num - 1]
        
        # 渲染页面为图像
        mat = fitz.Matrix(1.5, 1.5)  # 缩放因子
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # 转换为base64
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        doc.close()
        
        return jsonify({
            'success': True,
            'preview_image': f"data:image/png;base64,{img_base64}",
            'current_page': page_num,
            'total_pages': len(doc)
        })
        
    except ImportError:
        # 如果PyMuPDF不可用，返回基本信息
        return jsonify({
            'success': False,
            'error': '预览功能需要安装PyMuPDF库',
            'fallback': True
        })
    except Exception as e:
        return jsonify({'error': f'预览失败: {str(e)}'})

@app.route('/preview_chapters', methods=['POST'])
def preview_chapters():
    """预览章节"""
    filepath = request.json.get('filepath')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'})
    
    try:
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > 100:
            processor = LargeFilePDFProcessor()
        else:
            processor = PDFProcessor()
        
        chapters = processor.preview_chapters(filepath)
        
        return jsonify({
            'success': True,
            'chapters': chapters
        })
        
    except Exception as e:
        return jsonify({'error': f'章节预览失败: {str(e)}'})

@app.route('/split', methods=['POST'])
def split_pdf():
    """执行PDF分割 - 支持批量处理"""
    global processing_status
    
    if processing_status['is_processing']:
        return jsonify({'error': '正在处理中，请稍候'})
    
    data = request.json
    files_data = data.get('files', [])  # 批量文件数据
    custom_output_path = data.get('output_path', '')  # 自定义输出路径
    
    if not files_data:
        return jsonify({'error': '没有选择文件'})
    
    # 验证所有文件存在
    for file_data in files_data:
        filepath = file_data.get('filepath')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': f'文件不存在: {file_data.get("filename", "未知")}'})
    
    # 重置状态
    processing_status = {
        'is_processing': True,
        'progress': 0,
        'message': '开始批量处理...',
        'files': [],
        'error': None,
        'current_file': '',
        'total_files': len(files_data)
    }
    
    def process_in_background():
        global processing_status
        
        try:
            all_output_files = []
            total_files = len(files_data)
            
            # 确定输出基础目录
            if custom_output_path and os.path.isdir(custom_output_path):
                base_output_dir = custom_output_path
            else:
                base_output_dir = app.config['OUTPUT_FOLDER']
            
            # 处理每个文件
            for file_index, file_data in enumerate(files_data):
                filepath = file_data['filepath']
                filename = file_data['filename']
                mode = file_data.get('mode', 'pages')
                
                # 更新进度
                processing_status['current_file'] = filename
                processing_status['message'] = f'处理文件 {file_index + 1}/{total_files}: {filename}'
                processing_status['progress'] = int((file_index / total_files) * 90)
                
                # 创建文件专用输出目录
                file_base_name = os.path.splitext(filename)[0]
                output_dir = os.path.join(base_output_dir, f"{file_base_name}_split")
                os.makedirs(output_dir, exist_ok=True)
                
                # 选择处理器
                file_size_mb = file_data['file_size_mb']
                if file_size_mb > 100:
                    processor = LargeFilePDFProcessor(progress_callback=progress_callback)
                else:
                    processor = PDFProcessor(progress_callback=progress_callback)
                
                file_output = []
                
                # 根据模式执行分割
                if mode == 'pages':
                    pages_per_file = file_data.get('pages_per_file', 10)
                    file_output = processor.split_by_pages(filepath, output_dir, pages_per_file)
                    
                elif mode == 'size':
                    target_size_mb = file_data.get('target_size_mb', 5.0)
                    file_output = processor.split_by_size(filepath, output_dir, target_size_mb)
                    
                elif mode == 'range':
                    ranges_str = file_data.get('page_ranges', '')
                    ranges = parse_page_ranges(ranges_str)
                    file_output = processor.split_by_range(filepath, output_dir, ranges)
                    
                elif mode == 'chapter':
                    auto_detect = file_data.get('auto_detect', True)
                    custom_chapters = file_data.get('custom_chapters') if not auto_detect else None
                    file_output = processor.split_by_chapters(filepath, output_dir, auto_detect, custom_chapters)
                
                all_output_files.extend(file_output)
            
            # 最终处理
            processing_status['message'] = '正在创建下载包...'
            processing_status['progress'] = 95
            
            # 创建批量下载包
            if total_files == 1:
                zip_filename = os.path.splitext(files_data[0]['filename'])[0]
            else:
                zip_filename = f"batch_split_{total_files}_files"
            
            zip_path = create_download_package(all_output_files, zip_filename)
            
            # 完成状态
            status_message = f'批量分割完成！处理了{total_files}个文件，生成了{len(all_output_files)}个分割文件'
            
            processing_status.update({
                'is_processing': False,
                'progress': 100,
                'message': status_message,
                'files': [os.path.basename(f) for f in all_output_files],
                'download_path': zip_path,
                'processed_files': total_files,
                'output_files_count': len(all_output_files),
                'save_location': base_output_dir
            })
            
        except Exception as e:
            processing_status.update({
                'is_processing': False,
                'progress': 0,
                'message': '处理失败',
                'error': str(e)
            })
    
    # 在后台线程中处理
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': '开始处理'})

@app.route('/status')
def get_status():
    """获取处理状态"""
    return jsonify(processing_status)

@app.route('/download')
def download_files():
    """下载分割后的文件"""
    download_path = processing_status.get('download_path')
    
    if download_path and os.path.exists(download_path):
        return send_file(download_path, as_attachment=True, 
                        download_name=f"pdf_split_{os.path.basename(download_path)}")
    
    return jsonify({'error': '下载文件不存在'})

def parse_page_ranges(ranges_str):
    """解析页面范围字符串"""
    ranges = []
    if not ranges_str.strip():
        raise ValueError("页面范围不能为空")
    
    parts = ranges_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            start_page = int(start.strip())
            end_page = int(end.strip())
            if start_page <= 0 or end_page <= 0 or start_page > end_page:
                raise ValueError(f"无效的页面范围: {part}")
            ranges.append((start_page, end_page))
        else:
            page_num = int(part)
            if page_num <= 0:
                raise ValueError(f"无效的页面号: {part}")
            ranges.append((page_num, page_num))
    
    return ranges

def create_download_package(output_files, base_name):
    """创建下载包"""
    zip_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_split.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_files:
            if os.path.exists(file_path):
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname)
    
    return zip_path

def start_web_app():
    """启动Web应用"""
    print("=" * 60)
    print("       PDF分割工具 - 纯黑白简约版")
    print("=" * 60)
    print("✓ PDF分割功能")
    print("✓ 实时进度监控")
    print("=" * 60)
    
    # 自动打开浏览器
    def open_browser():
        webbrowser.open('http://127.0.0.1:5002')
    
    timer = threading.Timer(1.5, open_browser)
    timer.start()
    
    print("🚀 启动Web服务器...")
    print("📱 浏览器将自动打开: http://127.0.0.1:5002")
    print("💡 如果浏览器未自动打开，请手动访问上述地址")
    print("\n按 Ctrl+C 停止服务器")
    
    try:
        app.run(host='127.0.0.1', port=5002, debug=True)
    except KeyboardInterrupt:
        print("\n\n服务器已停止")

if __name__ == '__main__':
    start_web_app()