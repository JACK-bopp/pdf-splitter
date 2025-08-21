"""
PDF按章节智能分割模块
实现高效、无损的PDF章节分割功能
"""

import os
import re
import math
import time
from typing import List, Tuple, Optional, Dict, Callable
from PyPDF2 import PdfReader, PdfWriter


class PDFChapterSplitter:
    """PDF章节智能分割器"""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        初始化章节分割器
        
        Args:
            progress_callback: 进度回调函数
        """
        self.progress_callback = progress_callback
        self.chapter_patterns = [
            # 中文章节模式
            r'第[一二三四五六七八九十百千万\d]+章',
            r'第[一二三四五六七八九十百千万\d]+节',
            r'第[一二三四五六七八九十百千万\d]+部分',
            r'第[一二三四五六七八九十百千万\d]+篇',
            # 英文章节模式
            r'Chapter\s+\d+',
            r'CHAPTER\s+\d+',
            r'Section\s+\d+',
            r'SECTION\s+\d+',
            # 数字章节模式
            r'\d+\.\s*[^\d]',  # 1. 开头
            r'\d+\.\d+\s*[^\d]',  # 1.1 开头
            # 目录常见模式
            r'目\s*录',
            r'Table\s+of\s+Contents',
            r'CONTENTS',
        ]
        
    def _update_progress(self, current: int, total: int, message: str = ""):
        """更新进度"""
        if self.progress_callback:
            progress_info = {
                'extra_info': message
            }
            self.progress_callback(current, total, progress_info)
    
    def extract_text_from_page(self, page) -> str:
        """
        从页面提取文本
        
        Args:
            page: PDF页面对象
            
        Returns:
            页面文本内容
        """
        try:
            text = page.extract_text()
            # 清理文本，移除多余空白
            text = re.sub(r'\s+', ' ', text.strip())
            return text
        except Exception:
            return ""
    
    def detect_chapters(self, pdf_path: str) -> List[Dict]:
        """
        智能检测PDF中的章节
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            章节信息列表，包含页码和标题
        """
        chapters = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
                
                self._update_progress(0, total_pages, "正在分析章节结构...")
                
                for page_num in range(total_pages):
                    page = reader.pages[page_num]
                    text = self.extract_text_from_page(page)
                    
                    # 检查是否包含章节标题
                    for pattern in self.chapter_patterns:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            chapter_title = self._extract_chapter_title(text, match.start())
                            if chapter_title and len(chapter_title.strip()) > 0:
                                chapters.append({
                                    'page': page_num + 1,  # 页码从1开始
                                    'title': chapter_title.strip(),
                                    'pattern': pattern
                                })
                    
                    if page_num % 10 == 0:  # 每10页更新一次进度
                        self._update_progress(page_num + 1, total_pages, f"已分析 {page_num + 1}/{total_pages} 页")
                
                # 去重和排序
                chapters = self._deduplicate_chapters(chapters)
                chapters.sort(key=lambda x: x['page'])
                
                self._update_progress(total_pages, total_pages, f"章节分析完成，发现 {len(chapters)} 个章节")
                
                return chapters
                
        except Exception as e:
            raise Exception(f"章节检测失败: {str(e)}")
    
    def _extract_chapter_title(self, text: str, start_pos: int, max_length: int = 100) -> str:
        """
        提取章节标题
        
        Args:
            text: 页面文本
            start_pos: 章节标记开始位置
            max_length: 标题最大长度
            
        Returns:
            章节标题
        """
        # 找到章节标记后的文本
        remaining_text = text[start_pos:start_pos + max_length]
        
        # 查找第一个换行或句号作为标题结束
        end_markers = ['\n', '。', '.', ':', '：']
        end_pos = len(remaining_text)
        
        for marker in end_markers:
            pos = remaining_text.find(marker)
            if pos != -1 and pos > 0:
                end_pos = min(end_pos, pos)
        
        title = remaining_text[:end_pos].strip()
        return title
    
    def _deduplicate_chapters(self, chapters: List[Dict]) -> List[Dict]:
        """
        去除重复的章节
        
        Args:
            chapters: 原始章节列表
            
        Returns:
            去重后的章节列表
        """
        seen_titles = set()
        unique_chapters = []
        
        for chapter in chapters:
            # 使用页码和标题的组合来判断重复
            key = (chapter['page'], chapter['title'][:20])  # 只比较标题前20字符
            if key not in seen_titles:
                seen_titles.add(key)
                unique_chapters.append(chapter)
        
        return unique_chapters
    
    def split_by_chapters(self, input_path: str, output_dir: str, 
                         custom_chapters: Optional[List[Dict]] = None) -> List[str]:
        """
        按章节分割PDF
        
        Args:
            input_path: 输入PDF路径
            output_dir: 输出目录
            custom_chapters: 自定义章节列表，如果为None则自动检测
            
        Returns:
            生成的文件路径列表
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 检测或使用自定义章节
            if custom_chapters is None:
                chapters = self.detect_chapters(input_path)
            else:
                chapters = custom_chapters
            
            if not chapters:
                raise Exception("未检测到章节，请使用其他分割方式")
            
            output_files = []
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            with open(input_path, 'rb') as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
                
                # 添加结束页作为最后一个章节的结束
                chapter_ranges = []
                for i, chapter in enumerate(chapters):
                    start_page = chapter['page'] - 1  # 转换为0基索引
                    if i + 1 < len(chapters):
                        end_page = chapters[i + 1]['page'] - 2  # 下一章节前一页
                    else:
                        end_page = total_pages - 1  # 最后一页
                    
                    if start_page <= end_page:  # 确保范围有效
                        chapter_ranges.append({
                            'start': start_page,
                            'end': end_page,
                            'title': chapter['title']
                        })
                
                self._update_progress(0, len(chapter_ranges), "开始按章节分割...")
                
                for i, chapter_range in enumerate(chapter_ranges):
                    writer = PdfWriter()
                    
                    # 添加章节页面
                    for page_idx in range(chapter_range['start'], chapter_range['end'] + 1):
                        if page_idx < total_pages:
                            try:
                                writer.add_page(reader.pages[page_idx])
                            except Exception as e:
                                print(f"警告: 添加页面 {page_idx + 1} 时出错: {e}")
                                continue
                    
                    # 生成文件名，使用章节标题
                    safe_title = self._make_safe_filename(chapter_range['title'])
                    if len(safe_title) > 50:  # 限制文件名长度
                        safe_title = safe_title[:50] + "..."
                    
                    output_filename = f"{base_name}_第{i+1:02d}章_{safe_title}.pdf"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 写入文件
                    with open(output_path, 'wb') as output_file:
                        writer.write(output_file)
                    
                    output_files.append(output_path)
                    
                    # 更新进度
                    progress_msg = f"完成第 {i+1} 章: {chapter_range['title']}"
                    self._update_progress(i + 1, len(chapter_ranges), progress_msg)
                
                return output_files
                
        except Exception as e:
            raise Exception(f"按章节分割失败: {str(e)}")
    
    def _make_safe_filename(self, title: str) -> str:
        """
        创建安全的文件名
        
        Args:
            title: 原始标题
            
        Returns:
            安全的文件名
        """
        # 移除或替换不安全的字符
        unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        safe_title = title
        
        for char in unsafe_chars:
            safe_title = safe_title.replace(char, '_')
        
        # 移除多余的空格和下划线
        safe_title = re.sub(r'[_\s]+', '_', safe_title)
        safe_title = safe_title.strip('_')
        
        return safe_title
    
    def preview_chapters(self, pdf_path: str) -> List[Dict]:
        """
        预览检测到的章节
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            章节预览信息
        """
        chapters = self.detect_chapters(pdf_path)
        
        # 添加页面范围信息
        for i, chapter in enumerate(chapters):
            if i + 1 < len(chapters):
                chapter['end_page'] = chapters[i + 1]['page'] - 1
                chapter['page_count'] = chapter['end_page'] - chapter['page'] + 1
            else:
                # 需要获取总页数
                try:
                    with open(pdf_path, 'rb') as file:
                        reader = PdfReader(file)
                        total_pages = len(reader.pages)
                        chapter['end_page'] = total_pages
                        chapter['page_count'] = total_pages - chapter['page'] + 1
                except Exception:
                    chapter['end_page'] = chapter['page']
                    chapter['page_count'] = 1
        
        return chapters


def validate_pdf_integrity(original_path: str, split_files: List[str]) -> Dict:
    """
    验证PDF分割的完整性
    
    Args:
        original_path: 原始PDF路径
        split_files: 分割后的文件列表
        
    Returns:
        验证结果
    """
    result = {
        'success': False,
        'original_pages': 0,
        'split_pages': 0,
        'missing_pages': [],
        'errors': []
    }
    
    try:
        # 统计原始文件页数
        with open(original_path, 'rb') as file:
            original_reader = PdfReader(file)
            result['original_pages'] = len(original_reader.pages)
        
        # 统计分割文件总页数
        total_split_pages = 0
        for split_file in split_files:
            if os.path.exists(split_file):
                try:
                    with open(split_file, 'rb') as file:
                        split_reader = PdfReader(file)
                        total_split_pages += len(split_reader.pages)
                except Exception as e:
                    result['errors'].append(f"读取文件 {os.path.basename(split_file)} 失败: {e}")
            else:
                result['errors'].append(f"文件不存在: {os.path.basename(split_file)}")
        
        result['split_pages'] = total_split_pages
        
        # 检查页面完整性
        if result['original_pages'] == result['split_pages']:
            result['success'] = True
        else:
            missing_count = result['original_pages'] - result['split_pages']
            result['missing_pages'] = [f"可能缺失 {missing_count} 页"]
        
        return result
        
    except Exception as e:
        result['errors'].append(f"验证过程出错: {str(e)}")
        return result
