import os
import math
import gc
import time
import hashlib
import threading
import concurrent.futures
import multiprocessing
import json
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
from PyPDF2 import PdfReader, PdfWriter

# 尝试导入psutil，如果失败则提供备用方案
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# 新技术：导入高级PDF分析工具
try:
    import fitz  # PyMuPDF for advanced PDF analysis
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# 高级内存管理
try:
    import mmap
    HAS_MMAP = True
except ImportError:
    HAS_MMAP = False

from pdf_chapter_splitter import PDFChapterSplitter, validate_pdf_integrity


class PDFAnalyzer:
    """先进的PDF文件分析器 - 使用多种技术进行深度分析"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    def deep_analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        深度分析PDF文件，提供全面的文件信息
        使用多种技术并行分析，确保准确性和完整性
        """
        if pdf_path in self.analysis_cache:
            return self.analysis_cache[pdf_path]
        
        analysis_tasks = []
        
        # 任务1：基础PyPDF2分析
        analysis_tasks.append(
            self.thread_pool.submit(self._analyze_with_pypdf2, pdf_path)
        )
        
        # 任务2：高级PyMuPDF分析（如果可用）
        if HAS_PYMUPDF:
            analysis_tasks.append(
                self.thread_pool.submit(self._analyze_with_pymupdf, pdf_path)
            )
        
        # 任务3：文件系统级分析
        analysis_tasks.append(
            self.thread_pool.submit(self._analyze_file_system, pdf_path)
        )
        
        # 任务4：内存映射分析（大文件优化）
        if HAS_MMAP:
            analysis_tasks.append(
                self.thread_pool.submit(self._analyze_with_mmap, pdf_path)
            )
        
        # 收集所有分析结果
        results = {}
        for task in concurrent.futures.as_completed(analysis_tasks):
            try:
                result = task.result()
                results.update(result)
            except Exception as e:
                results[f'error_{task}'] = str(e)
        
        # 合成最终分析报告
        final_analysis = self._synthesize_analysis(results)
        self.analysis_cache[pdf_path] = final_analysis
        
        return final_analysis
    
    def _analyze_with_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """使用PyPDF2进行基础分析"""
        try:
            reader = PdfReader(pdf_path)
            return {
                'pypdf2_pages': len(reader.pages),
                'pypdf2_metadata': reader.metadata,
                'pypdf2_encrypted': reader.is_encrypted,
                'pypdf2_form_fields': reader.get_form_text_fields() if hasattr(reader, 'get_form_text_fields') else None,
                'pypdf2_status': 'success'
            }
        except Exception as e:
            return {'pypdf2_status': 'failed', 'pypdf2_error': str(e)}
    
    def _analyze_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """使用PyMuPDF进行高级分析"""
        try:
            doc = fitz.open(pdf_path)
            analysis = {
                'pymupdf_pages': doc.page_count,
                'pymupdf_metadata': doc.metadata,
                'pymupdf_toc': doc.get_toc(),
                'pymupdf_page_sizes': [],
                'pymupdf_text_info': [],
                'pymupdf_image_info': [],
                'pymupdf_status': 'success'
            }
            
            # 分析前几页获取样本信息
            sample_pages = min(5, doc.page_count)
            for page_num in range(sample_pages):
                page = doc[page_num]
                analysis['pymupdf_page_sizes'].append({
                    'page': page_num + 1,
                    'width': page.rect.width,
                    'height': page.rect.height
                })
                
                # 文本分析
                text_blocks = page.get_text("dict")
                analysis['pymupdf_text_info'].append({
                    'page': page_num + 1,
                    'text_length': len(page.get_text()),
                    'block_count': len(text_blocks.get('blocks', []))
                })
                
                # 图像分析
                images = page.get_images()
                analysis['pymupdf_image_info'].append({
                    'page': page_num + 1,
                    'image_count': len(images)
                })
            
            doc.close()
            return analysis
        except Exception as e:
            return {'pymupdf_status': 'failed', 'pymupdf_error': str(e)}
    
    def _analyze_file_system(self, pdf_path: str) -> Dict[str, Any]:
        """文件系统级分析"""
        try:
            stat = os.stat(pdf_path)
            return {
                'file_size_bytes': stat.st_size,
                'file_size_mb': stat.st_size / (1024 * 1024),
                'creation_time': stat.st_ctime,
                'modification_time': stat.st_mtime,
                'file_accessible': os.access(pdf_path, os.R_OK),
                'file_writable': os.access(pdf_path, os.W_OK),
                'filesystem_status': 'success'
            }
        except Exception as e:
            return {'filesystem_status': 'failed', 'filesystem_error': str(e)}
    
    def _analyze_with_mmap(self, pdf_path: str) -> Dict[str, Any]:
        """使用内存映射进行大文件分析"""
        try:
            with open(pdf_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # 检查PDF头部
                    pdf_header = mm[:10].decode('latin-1', errors='ignore')
                    
                    # 快速扫描PDF结构
                    xref_count = mm.count(b'xref')
                    obj_count = mm.count(b'obj')
                    stream_count = mm.count(b'stream')
                    
                    return {
                        'mmap_pdf_header': pdf_header,
                        'mmap_xref_count': xref_count,
                        'mmap_obj_count': obj_count,
                        'mmap_stream_count': stream_count,
                        'mmap_status': 'success'
                    }
        except Exception as e:
            return {'mmap_status': 'failed', 'mmap_error': str(e)}
    
    def _synthesize_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """合成分析结果，生成最终报告"""
        synthesis = {
            'analysis_timestamp': time.time(),
            'analysis_methods_used': [],
            'confidence_score': 0,
            'issues_detected': [],
            'recommendations': []
        }
        
        # 确定成功的分析方法
        if results.get('pypdf2_status') == 'success':
            synthesis['analysis_methods_used'].append('PyPDF2')
            synthesis['confidence_score'] += 30
        
        if results.get('pymupdf_status') == 'success':
            synthesis['analysis_methods_used'].append('PyMuPDF')
            synthesis['confidence_score'] += 40
        
        if results.get('filesystem_status') == 'success':
            synthesis['analysis_methods_used'].append('FileSystem')
            synthesis['confidence_score'] += 20
        
        if results.get('mmap_status') == 'success':
            synthesis['analysis_methods_used'].append('MemoryMapping')
            synthesis['confidence_score'] += 10
        
        # 页面数量一致性检查
        page_counts = []
        if 'pypdf2_pages' in results:
            page_counts.append(results['pypdf2_pages'])
        if 'pymupdf_pages' in results:
            page_counts.append(results['pymupdf_pages'])
        
        if page_counts and len(set(page_counts)) > 1:
            synthesis['issues_detected'].append('页面数量不一致')
            synthesis['confidence_score'] -= 20
        
        # 文件大小检查
        if results.get('file_size_mb', 0) > 100:
            synthesis['recommendations'].append('建议使用大文件处理模式')
        
        # 加密检查
        if results.get('pypdf2_encrypted', False):
            synthesis['issues_detected'].append('文件已加密')
            synthesis['recommendations'].append('需要提供密码解密')
        
        # 合并所有结果
        synthesis.update(results)
        
        return synthesis


class PDFProcessor:
    """PDF处理器类，专为GB级大文件优化的PDF分割工具"""
    
    def __init__(self, progress_callback: Optional[Callable] = None, memory_limit_mb: int = 512):
        """
        初始化PDF处理器
        
        Args:
            progress_callback: 进度回调函数，接收(current, total, status_info)参数
            memory_limit_mb: 内存使用限制（MB），超过时触发垃圾回收
        """
        self.progress_callback = progress_callback
        self.memory_limit_mb = memory_limit_mb
        self.start_time = None
        self.processed_pages = 0
        self.total_pages = 0
        self.chapter_splitter = PDFChapterSplitter(progress_callback)
        self.analyzer = PDFAnalyzer()
        self.processing_lock = threading.Lock()
        self.error_recovery_enabled = True
        
    def _monitor_memory(self) -> None:
        """智能内存监控和管理"""
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 获取系统内存信息
                system_memory = psutil.virtual_memory()
                available_mb = system_memory.available / 1024 / 1024
                
                # 多级内存管理策略
                if memory_mb > self.memory_limit_mb * 0.8:  # 达到80%阈值
                    # 轻度垃圾回收
                    gc.collect()
                    
                if memory_mb > self.memory_limit_mb:  # 达到100%阈值
                    # 强制垃圾回收
                    gc.collect()
                    gc.collect()  # 二次回收
                    
                if available_mb < 200:  # 系统可用内存少于200MB
                    # 紧急内存释放
                    import ctypes
                    if hasattr(ctypes, 'windll'):  # Windows
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                    gc.collect()
                    gc.collect()
                    gc.collect()  # 三次强制回收
            except Exception:
                # 如果psutil出错，只执行基本垃圾回收
                gc.collect()
        else:
            # 没有psutil时的简单内存管理
            gc.collect()
            
    def _update_progress(self, current: int, total: int, extra_info: str = "") -> None:
        """更新进度并提供详细状态信息"""
        if self.progress_callback:
            # 计算处理速度和预估剩余时间
            if self.start_time:
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 0 and current > 0:
                    if HAS_PSUTIL:
                        try:
                            pages_per_second = current / elapsed_time
                            remaining_pages = total - current
                            eta_seconds = remaining_pages / pages_per_second if pages_per_second > 0 else 0
                            
                            # 内存使用情况
                            process = psutil.Process()
                            memory_mb = process.memory_info().rss / 1024 / 1024
                            
                            status_info = {
                                'pages_per_second': pages_per_second,
                                'eta_seconds': eta_seconds,
                                'memory_mb': memory_mb,
                                'extra_info': extra_info
                            }
                            
                            self.progress_callback(current, total, status_info)
                        except Exception:
                            self.progress_callback(current, total, {'extra_info': extra_info})
                    else:
                        self.progress_callback(current, total, {'extra_info': extra_info})
                else:
                    self.progress_callback(current, total, {'extra_info': extra_info})
            else:
                self.progress_callback(current, total, {'extra_info': extra_info})

    def split_by_pages(self, pdf_path: str, output_dir: str, pages_per_file: int = 10) -> List[str]:
        """
        增强版按页数分割PDF文件 - 集成先进分析和错误恢复
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            pages_per_file: 每个文件包含的页数
            
        Returns:
            List[str]: 生成的文件路径列表
        """
        self.start_time = time.time()
        output_files = []
        
        with self.processing_lock:
            try:
                # 第1步：深度文件分析
                self._update_progress(0, 100, "正在进行深度文件分析...")
                analysis_result = self.analyzer.deep_analyze_pdf(pdf_path)
                
                # 检查分析结果
                if analysis_result.get('confidence_score', 0) < 50:
                    issues = ', '.join(analysis_result.get('issues_detected', []))
                    raise RuntimeError(f"文件分析置信度不足: {issues}")
                
                # 根据分析结果优化处理策略
                if analysis_result.get('file_size_mb', 0) > 100:
                    self._update_progress(5, 100, "检测到大文件，启用优化模式...")
                    return self._split_large_file_optimized(pdf_path, output_dir, pages_per_file, analysis_result)
                
                # 第2步：多方式文件读取
                self._update_progress(10, 100, "使用多种方式读取文件...")
                reader = None
                
                # 尝试多种读取方式
                for attempt in range(3):
                    try:
                        if attempt == 0:
                            # 标准方式
                            reader = PdfReader(pdf_path)
                        elif attempt == 1:
                            # 严格模式
                            reader = PdfReader(pdf_path, strict=False)
                        else:
                            # 使用内存映射
                            if HAS_MMAP:
                                with open(pdf_path, 'rb') as f:
                                    reader = PdfReader(f)
                            else:
                                reader = PdfReader(pdf_path, strict=False)
                        break
                    except Exception as e:
                        if attempt == 2:  # 最后一次尝试
                            raise RuntimeError(f"多次尝试读取文件失败: {str(e)}")
                        continue
                
                total_pages = len(reader.pages)
                self.total_pages = total_pages
                
                if total_pages == 0:
                    raise ValueError("PDF文件没有页面")
                
                # 验证分析结果的一致性
                expected_pages = analysis_result.get('pypdf2_pages') or analysis_result.get('pymupdf_pages')
                if expected_pages and abs(total_pages - expected_pages) > 0:
                    self._update_progress(15, 100, f"警告：页面数量不一致，继续处理...")
                
                # 第3步：智能分割策略
                num_files = math.ceil(total_pages / pages_per_file)
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                self._update_progress(20, 100, f"开始分割文件：{total_pages}页 -> {num_files}个文件")
                
                # 第4步：并行分割处理
                if num_files > 4 and total_pages > 100:
                    # 大量文件使用并行处理
                    output_files = self._parallel_split_pages(reader, output_dir, base_name, pages_per_file, total_pages)
                else:
                    # 小文件使用顺序处理
                    output_files = self._sequential_split_pages(reader, output_dir, base_name, pages_per_file, total_pages)
                
                # 第5步：完整性验证
                self._update_progress(95, 100, "正在验证文件完整性...")
                validation_result = self._validate_split_results(pdf_path, output_files)
                
                if not validation_result['success']:
                    if self.error_recovery_enabled:
                        self._update_progress(97, 100, "检测到问题，启动错误恢复...")
                        output_files = self._recover_from_split_error(pdf_path, output_dir, pages_per_file, validation_result)
                    else:
                        raise RuntimeError(f"分割验证失败: {validation_result['error']}")
                
                self._update_progress(100, 100, f"分割完成：生成{len(output_files)}个文件")
                
            except Exception as e:
                if self.error_recovery_enabled:
                    # 尝试错误恢复
                    try:
                        self._update_progress(50, 100, "尝试错误恢复...")
                        return self._fallback_split_method(pdf_path, output_dir, pages_per_file)
                    except Exception as recovery_error:
                        raise RuntimeError(f"分割失败且恢复失败: 原错误={str(e)}, 恢复错误={str(recovery_error)}")
                else:
                    raise RuntimeError(f"按页数分割失败: {str(e)}")
        
        return output_files
    
    def _split_large_file_optimized(self, pdf_path: str, output_dir: str, pages_per_file: int, analysis: Dict[str, Any]) -> List[str]:
        """大文件优化分割"""
        output_files = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 使用流式处理
            with open(pdf_path, 'rb') as source_file:
                reader = PdfReader(source_file)
                total_pages = len(reader.pages)
                
                # 分批处理以减少内存使用
                batch_size = max(1, min(pages_per_file, 10))  # 限制批次大小
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                for file_index in range(0, math.ceil(total_pages / pages_per_file)):
                    start_page = file_index * pages_per_file
                    end_page = min(start_page + pages_per_file, total_pages)
                    
                    # 创建临时文件
                    temp_path = os.path.join(temp_dir, f"temp_{file_index}.pdf")
                    
                    with PdfWriter() as writer:
                        for page_idx in range(start_page, end_page):
                            writer.add_page(reader.pages[page_idx])
                            self.processed_pages += 1
                            
                            if self.processed_pages % batch_size == 0:
                                progress = int((self.processed_pages / total_pages) * 80) + 20
                                self._update_progress(progress, 100, f"大文件处理: {self.processed_pages}/{total_pages}")
                                self._monitor_memory()
                        
                        # 写入临时文件
                        with open(temp_path, 'wb') as temp_file:
                            writer.write(temp_file)
                    
                    # 移动到最终位置
                    final_name = f"{base_name}_part_{file_index + 1:03d}.pdf"
                    final_path = os.path.join(output_dir, final_name)
                    
                    os.rename(temp_path, final_path)
                    output_files.append(final_path)
                    
                    # 验证文件
                    if not self._verify_output_file(final_path):
                        raise RuntimeError(f"大文件分割验证失败: {final_name}")
        
        finally:
            # 清理临时目录
            try:
                for temp_file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, temp_file))
                os.rmdir(temp_dir)
            except:
                pass
        
        return output_files
    
    def _parallel_split_pages(self, reader: PdfReader, output_dir: str, base_name: str, pages_per_file: int, total_pages: int) -> List[str]:
        """并行分割处理"""
        output_files = []
        num_files = math.ceil(total_pages / pages_per_file)
        
        def split_batch(file_index: int) -> str:
            start_page = file_index * pages_per_file
            end_page = min(start_page + pages_per_file, total_pages)
            
            writer = PdfWriter()
            for page_idx in range(start_page, end_page):
                writer.add_page(reader.pages[page_idx])
            
            output_filename = f"{base_name}_part_{file_index + 1:03d}.pdf"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            return output_path
        
        # 使用线程池并行处理
        max_workers = min(4, multiprocessing.cpu_count())
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(split_batch, i): i for i in range(num_files)}
            
            for future in concurrent.futures.as_completed(future_to_index):
                file_index = future_to_index[future]
                try:
                    output_path = future.result()
                    output_files.append(output_path)
                    
                    # 更新进度
                    progress = int((len(output_files) / num_files) * 70) + 20
                    self._update_progress(progress, 100, f"并行处理: {len(output_files)}/{num_files}")
                    
                except Exception as e:
                    raise RuntimeError(f"并行分割第{file_index + 1}个文件失败: {str(e)}")
        
        # 按文件索引排序
        output_files.sort()
        return output_files
    
    def _sequential_split_pages(self, reader: PdfReader, output_dir: str, base_name: str, pages_per_file: int, total_pages: int) -> List[str]:
        """顺序分割处理"""
        output_files = []
        num_files = math.ceil(total_pages / pages_per_file)
        
        for file_index in range(num_files):
            start_page = file_index * pages_per_file
            end_page = min(start_page + pages_per_file, total_pages)
            
            writer = PdfWriter()
            
            for page_index in range(start_page, end_page):
                writer.add_page(reader.pages[page_index])
                self.processed_pages += 1
                
                # 更新进度
                progress = int((self.processed_pages / total_pages) * 70) + 20
                self._update_progress(progress, 100, f"顺序处理: {self.processed_pages}/{total_pages}")
                
                # 内存监控
                if self.processed_pages % 10 == 0:
                    self._monitor_memory()
            
            # 写入文件
            output_filename = f"{base_name}_part_{file_index + 1:03d}.pdf"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            output_files.append(output_path)
            
            # 验证生成的文件
            if not self._verify_output_file(output_path):
                raise RuntimeError(f"生成的文件验证失败: {output_filename}")
            
            # 清理内存
            del writer
            gc.collect()
        
        return output_files
    
    def _validate_split_results(self, original_path: str, split_files: List[str]) -> Dict[str, Any]:
        """验证分割结果"""
        try:
            original_reader = PdfReader(original_path)
            original_pages = len(original_reader.pages)
            
            total_split_pages = 0
            for split_file in split_files:
                if not os.path.exists(split_file):
                    return {'success': False, 'error': f'文件不存在: {split_file}'}
                
                split_reader = PdfReader(split_file)
                total_split_pages += len(split_reader.pages)
            
            if original_pages != total_split_pages:
                return {'success': False, 'error': f'页面数量不匹配: 原文件{original_pages}页，分割文件共{total_split_pages}页'}
            
            return {'success': True, 'original_pages': original_pages, 'split_pages': total_split_pages}
            
        except Exception as e:
            return {'success': False, 'error': f'验证过程出错: {str(e)}'}
    
    def _recover_from_split_error(self, pdf_path: str, output_dir: str, pages_per_file: int, validation_result: Dict[str, Any]) -> List[str]:
        """错误恢复机制"""
        self._update_progress(98, 100, "执行错误恢复...")
        
        # 清理可能损坏的文件
        for file_path in os.listdir(output_dir):
            full_path = os.path.join(output_dir, file_path)
            if os.path.isfile(full_path) and file_path.endswith('.pdf'):
                try:
                    os.remove(full_path)
                except:
                    pass
        
        # 使用更保守的方法重新分割
        return self._fallback_split_method(pdf_path, output_dir, pages_per_file)
    
    def _fallback_split_method(self, pdf_path: str, output_dir: str, pages_per_file: int) -> List[str]:
        """备用分割方法"""
        output_files = []
        
        try:
            # 使用最简单最可靠的方法
            reader = PdfReader(pdf_path, strict=False)
            total_pages = len(reader.pages)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            current_page = 0
            file_index = 1
            
            while current_page < total_pages:
                writer = PdfWriter()
                pages_added = 0
                
                # 一页一页添加，更安全
                while current_page < total_pages and pages_added < pages_per_file:
                    try:
                        writer.add_page(reader.pages[current_page])
                        pages_added += 1
                        current_page += 1
                    except Exception:
                        current_page += 1  # 跳过问题页面
                        continue
                
                if pages_added > 0:
                    output_filename = f"{base_name}_fallback_{file_index:03d}.pdf"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    with open(output_path, 'wb') as output_file:
                        writer.write(output_file)
                    
                    output_files.append(output_path)
                    file_index += 1
                
                del writer
                gc.collect()
            
            return output_files
            
        except Exception as e:
            raise RuntimeError(f"备用方法也失败: {str(e)}")
    
    def get_pdf_info(self, pdf_path: str) -> Tuple[int, float, Dict[str, Any]]:
        """
        获取PDF文件详细信息 - 使用先进分析技术
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            Tuple[int, float, Dict]: (页数, 文件大小MB, 详细信息字典)
        """
        try:
            # 使用深度分析获取详细信息
            analysis_result = self.analyzer.deep_analyze_pdf(pdf_path)
            
            # 提取基础信息
            page_count = analysis_result.get('pypdf2_pages', 0) or analysis_result.get('pymupdf_pages', 0)
            file_size_mb = analysis_result.get('file_size_mb', 0)
            
            # 构建详细信息
            detailed_info = {
                'is_large_file': file_size_mb > 100,
                'confidence_score': analysis_result.get('confidence_score', 0),
                'analysis_methods': analysis_result.get('analysis_methods_used', []),
                'issues_detected': analysis_result.get('issues_detected', []),
                'recommendations': analysis_result.get('recommendations', []),
                'encrypted': analysis_result.get('pypdf2_encrypted', False),
                'has_metadata': bool(analysis_result.get('pypdf2_metadata') or analysis_result.get('pymupdf_metadata')),
                'has_toc': bool(analysis_result.get('pymupdf_toc', [])),
                'page_sizes': analysis_result.get('pymupdf_page_sizes', []),
                'text_info': analysis_result.get('pymupdf_text_info', []),
                'image_info': analysis_result.get('pymupdf_image_info', []),
                'avg_page_size_kb': (file_size_mb * 1024 / page_count) if page_count > 0 else 0,
                'pdf_version': analysis_result.get('mmap_pdf_header', ''),
                'object_count': analysis_result.get('mmap_obj_count', 0),
                'stream_count': analysis_result.get('mmap_stream_count', 0)
            }
            
            return page_count, file_size_mb, detailed_info
            
        except Exception as e:
            # 备用基础信息获取
            try:
                reader = PdfReader(pdf_path)
                page_count = len(reader.pages)
                file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                
                detailed_info = {
                    'is_large_file': file_size_mb > 100,
                    'confidence_score': 30,
                    'analysis_methods': ['PyPDF2基础'],
                    'issues_detected': ['深度分析失败'],
                    'recommendations': ['建议检查文件完整性'],
                    'encrypted': reader.is_encrypted,
                    'has_metadata': bool(reader.metadata),
                    'has_toc': False,
                    'avg_page_size_kb': (file_size_mb * 1024 / page_count) if page_count > 0 else 0,
                    'error': str(e)
                }
                
                return page_count, file_size_mb, detailed_info
                
            except Exception as basic_error:
                raise RuntimeError(f"文件信息获取失败: {str(basic_error)}")

    def split_by_size(self, pdf_path: str, output_dir: str, target_size_mb: float = 5.0) -> List[str]:
        """
        增强版按文件大小分割PDF - 集成先进分析和错误恢复
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            target_size_mb: 目标文件大小（MB）
            
        Returns:
            List[str]: 生成的文件路径列表
        """
        self.start_time = time.time()
        output_files = []
        
        with self.processing_lock:
            try:
                # 第1步：深度文件分析
                self._update_progress(0, 100, "正在进行深度文件分析...")
                analysis_result = self.analyzer.deep_analyze_pdf(pdf_path)
                
                # 从分析结果获取可靠的页面数和文件大小
                total_pages = analysis_result.get('pypdf2_pages', 0) or analysis_result.get('pymupdf_pages', 0)
                file_size_mb = analysis_result.get('file_size_mb', 0)
                
                # 如果分析失败，使用备用方法
                if not total_pages or not file_size_mb:
                    reader = PdfReader(pdf_path)
                    total_pages = len(reader.pages)
                    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                
                # 确保数据类型正确
                total_pages = int(total_pages)
                file_size_mb = float(file_size_mb)
                target_size_mb = float(target_size_mb)
                
                self.total_pages = total_pages
                
                if total_pages == 0:
                    raise ValueError("PDF文件没有页面")
                
                if file_size_mb <= 0:
                    raise ValueError("无法获取有效的文件大小")
                
                # 第2步：智能大小估算
                self._update_progress(10, 100, "计算最优分割策略...")
                avg_page_size_mb = file_size_mb / total_pages
                estimated_pages_per_file = max(1, int(target_size_mb / avg_page_size_mb))
                
                # 安全检查：避免单个文件页面过多
                max_pages_per_file = min(estimated_pages_per_file, 1000)
                
                self._update_progress(20, 100, f"预计每文件{max_pages_per_file}页，目标大小{target_size_mb}MB")
                
                # 第3步：执行分割
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                current_page = 0
                file_index = 1
                
                # 重新读取文件进行分割
                reader = PdfReader(pdf_path)
                
                while current_page < total_pages:
                    writer = PdfWriter()
                    pages_in_current_file = 0
                    
                    # 动态添加页面直到达到目标大小
                    while current_page < total_pages and pages_in_current_file < max_pages_per_file:
                        try:
                            writer.add_page(reader.pages[current_page])
                            pages_in_current_file += 1
                            current_page += 1
                            self.processed_pages += 1
                            
                            # 更新进度
                            progress = int((self.processed_pages / total_pages) * 80) + 20
                            progress_msg = f"按大小分割: {self.processed_pages}/{total_pages} 页"
                            self._update_progress(progress, 100, progress_msg)
                            
                            # 内存监控
                            if self.processed_pages % 10 == 0:
                                self._monitor_memory()
                                
                        except Exception as page_error:
                            # 跳过问题页面，继续处理
                            current_page += 1
                            continue
                    
                    # 第4步：写入文件
                    if pages_in_current_file > 0:
                        output_filename = f"{base_name}_size_{file_index:03d}.pdf"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        with open(output_path, 'wb') as output_file:
                            writer.write(output_file)
                        
                        output_files.append(output_path)
                        
                        # 验证生成的文件
                        if not self._verify_output_file(output_path):
                            raise RuntimeError(f"生成的文件验证失败: {output_filename}")
                        
                        file_index += 1
                    
                    del writer
                    gc.collect()
                
                # 第5步：完整性验证
                self._update_progress(95, 100, "验证分割结果...")
                validation_result = self._validate_split_results(pdf_path, output_files)
                
                if not validation_result['success'] and self.error_recovery_enabled:
                    self._update_progress(97, 100, "检测到问题，启动错误恢复...")
                    return self._recover_from_split_error(pdf_path, output_dir, max_pages_per_file, validation_result)
                
                self._update_progress(100, 100, f"按大小分割完成：生成{len(output_files)}个文件")
                
            except Exception as e:
                if self.error_recovery_enabled:
                    try:
                        self._update_progress(50, 100, "尝试备用分割方法...")
                        return self._fallback_split_by_size(pdf_path, output_dir, target_size_mb)
                    except Exception as recovery_error:
                        raise RuntimeError(f"按大小分割失败且恢复失败: 原错误={str(e)}, 恢复错误={str(recovery_error)}")
                else:
                    raise RuntimeError(f"按大小分割失败: {str(e)}")
        
        return output_files
    
    def _fallback_split_by_size(self, pdf_path: str, output_dir: str, target_size_mb: float) -> List[str]:
        """备用按大小分割方法"""
        output_files = []
        
        try:
            reader = PdfReader(pdf_path, strict=False)
            total_pages = len(reader.pages)
            
            # 使用保守的估算
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            avg_page_size_mb = file_size_mb / total_pages if total_pages > 0 else 1.0
            pages_per_file = max(1, min(int(target_size_mb / avg_page_size_mb), 100))  # 限制最大页数
            
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            current_page = 0
            file_index = 1
            
            while current_page < total_pages:
                writer = PdfWriter()
                pages_added = 0
                
                while current_page < total_pages and pages_added < pages_per_file:
                    try:
                        writer.add_page(reader.pages[current_page])
                        pages_added += 1
                        current_page += 1
                    except Exception:
                        current_page += 1  # 跳过问题页面
                        continue
                
                if pages_added > 0:
                    output_filename = f"{base_name}_fallback_size_{file_index:03d}.pdf"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    with open(output_path, 'wb') as output_file:
                        writer.write(output_file)
                    
                    output_files.append(output_path)
                    file_index += 1
                
                del writer
                gc.collect()
            
            return output_files
            
        except Exception as e:
            raise RuntimeError(f"备用按大小分割方法失败: {str(e)}")

    def split_by_range(self, pdf_path: str, output_dir: str, page_ranges: List[Tuple[int, int]]) -> List[str]:
        """
        按页面范围分割PDF
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            page_ranges: 页面范围列表，如[(1, 10), (11, 20)]
            
        Returns:
            List[str]: 生成的文件路径列表
        """
        self.start_time = time.time()
        output_files = []
        
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            self.total_pages = sum(end - start + 1 for start, end in page_ranges)
            
            if total_pages == 0:
                raise ValueError("PDF文件没有页面")
            
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            for range_index, (start_page, end_page) in enumerate(page_ranges):
                # 验证页面范围
                if start_page < 1 or end_page > total_pages or start_page > end_page:
                    raise ValueError(f"无效的页面范围: {start_page}-{end_page}")
                
                writer = PdfWriter()
                
                # 添加指定范围的页面
                for page_index in range(start_page - 1, end_page):  # 转换为0基索引
                    writer.add_page(reader.pages[page_index])
                    self.processed_pages += 1
                    
                    # 更新进度
                    progress_msg = f"范围分割: {self.processed_pages}/{self.total_pages} 页"
                    self._update_progress(self.processed_pages, self.total_pages, progress_msg)
                    
                    # 内存监控
                    if self.processed_pages % 10 == 0:
                        self._monitor_memory()
                
                # 写入文件
                output_filename = f"{base_name}_range_{start_page}-{end_page}.pdf"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'wb') as output_file:
                    writer.write(output_file)
                
                output_files.append(output_path)
                
                # 验证生成的文件
                if not self._verify_output_file(output_path):
                    raise RuntimeError(f"生成的文件验证失败: {output_filename}")
                
                del writer
                gc.collect()
                
        except Exception as e:
            raise RuntimeError(f"按范围分割失败: {str(e)}")
        
        return output_files

    def split_by_chapters(self, pdf_path: str, output_dir: str, 
                         auto_detect: bool = True, custom_chapters: Optional[List[str]] = None) -> List[str]:
        """
        按章节分割PDF
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            auto_detect: 是否自动检测章节
            custom_chapters: 自定义章节标题列表
            
        Returns:
            List[str]: 生成的文件路径列表
        """
        try:
            return self.chapter_splitter.split_by_chapters(
                pdf_path, output_dir, auto_detect, custom_chapters
            )
        except Exception as e:
            raise RuntimeError(f"按章节分割失败: {str(e)}")

    def _verify_output_file(self, file_path: str) -> bool:
        """验证输出文件的完整性"""
        try:
            if not os.path.exists(file_path):
                return False
            
            # 检查文件大小
            if os.path.getsize(file_path) == 0:
                return False
            
            # 尝试打开PDF文件
            reader = PdfReader(file_path)
            if len(reader.pages) == 0:
                return False
            
            return True
        except Exception:
            return False


class LargeFilePDFProcessor(PDFProcessor):
    """针对大文件优化的PDF处理器"""
    
    def __init__(self, progress_callback: Optional[Callable] = None, memory_limit_mb: int = 256):
        super().__init__(progress_callback, memory_limit_mb)
        
    def split_by_pages(self, pdf_path: str, output_dir: str, pages_per_file: int = 10) -> List[str]:
        """大文件优化的按页分割"""
        self.start_time = time.time()
        output_files = []
        
        try:
            # 使用更小的批次处理大文件
            batch_size = min(pages_per_file, 5)  # 限制批处理大小
            
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            self.total_pages = total_pages
            
            if total_pages == 0:
                raise ValueError("PDF文件没有页面")
            
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            current_page = 0
            file_index = 1
            
            while current_page < total_pages:
                writer = PdfWriter()
                pages_in_file = 0
                
                # 小批次添加页面
                while current_page < total_pages and pages_in_file < pages_per_file:
                    writer.add_page(reader.pages[current_page])
                    pages_in_file += 1
                    current_page += 1
                    self.processed_pages += 1
                    
                    # 更新进度和内存监控
                    if self.processed_pages % 5 == 0:  # 更频繁的监控
                        progress_msg = f"大文件分割: {self.processed_pages}/{total_pages} 页"
                        self._update_progress(self.processed_pages, total_pages, progress_msg)
                        self._monitor_memory()
                
                # 写入文件
                output_filename = f"{base_name}_part_{file_index:03d}.pdf"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'wb') as output_file:
                    writer.write(output_file)
                
                output_files.append(output_path)
                
                # 验证生成的文件
                if not self._verify_output_file(output_path):
                    raise RuntimeError(f"生成的文件验证失败: {output_filename}")
                
                file_index += 1
                del writer
                gc.collect()
                
        except Exception as e:
            raise RuntimeError(f"大文件按页数分割失败: {str(e)}")
        
        return output_files


def comprehensive_integrity_check(original_path: str, split_files: List[str]) -> dict:
    """全面的完整性检查"""
    try:
        original_reader = PdfReader(original_path)
        original_pages = len(original_reader.pages)
        
        total_split_pages = 0
        for split_file in split_files:
            if os.path.exists(split_file):
                split_reader = PdfReader(split_file)
                total_split_pages += len(split_reader.pages)
        
        return {
            'original_pages': original_pages,
            'split_pages': total_split_pages,
            'pages_match': original_pages == total_split_pages,
            'all_files_exist': all(os.path.exists(f) for f in split_files),
            'check_passed': original_pages == total_split_pages and all(os.path.exists(f) for f in split_files)
        }
    except Exception as e:
        return {
            'error': str(e),
            'check_passed': False
        }


def create_integrity_report(check_result: dict) -> str:
    """创建完整性检查报告"""
    if check_result.get('check_passed', False):
        return "✓ PDF分割完整性检查通过"
    else:
        return f"✗ PDF分割完整性检查失败: {check_result.get('error', '页面数量不匹配')}"
