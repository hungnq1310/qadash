import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
from pathlib import Path
from PIL import Image

# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader, TextLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any, Dict

# Transformers for Qwen2-VL
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

class Qwen2VLLLM:
    def __init__(self):
        super().__init__()
        print("🔄 Đang tải mô hình...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        print("✅ Mô hình đã sẵn sàng!")
    
    @property
    def _llm_type(self) -> str:
        return "qwen2vl"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=None,
                padding=True,
                return_tensors="pt"
            )
            for key in inputs:
                inputs[key] = inputs[key].to(self.model.device)
                
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
            
        except Exception as e:
            return f"Lỗi: {str(e)}"
    
    def analyze_image_with_text(self, image_path: str, question: str) -> str:
        try:
            image = Image.open(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"Phân tích ảnh này và trả lời: {question}. Trả lời bằng tiếng Việt."}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
            
        except Exception as e:
            return f"Lỗi khi phân tích ảnh: {str(e)}"

    def classify_question(self, question: str) -> str:
        """Phân loại câu hỏi để xác định có cần phân tích ảnh hay không"""
        
        # Từ khóa liên quan đến ảnh/biểu đồ/dashboard
        image_keywords = [
            'ảnh', 'hình', 'biểu đồ', 'chart', 'dashboard', 'màn hình', 'giao diện',
            'nhìn thấy', 'hiển thị', 'trong ảnh', 'trên ảnh', 'ở ảnh', 'từ ảnh',
            'màu sắc', 'đồ thị', 'visualization', 'visual', 'screen', 'interface',
            'layout', 'design', 'ui', 'ux', 'screenshot'
        ]
        
        # Từ khóa liên quan đến dữ liệu số
        data_keywords = [
            'số liệu', 'dữ liệu', 'data', 'volume', 'target', 'doanh thu', 'revenue',
            'tổng', 'sum', 'average', 'trung bình', 'thống kê', 'phân tích số',
            'bu', 'đơn vị', 'khối lượng', 'percentage', 'phần trăm', '%',
            'tính toán', 'calculate', 'count', 'đếm', 'so sánh', 'compare'
        ]
        
        question_lower = question.lower()
        
        # Đếm số từ khóa xuất hiện
        image_score = sum(1 for keyword in image_keywords if keyword in question_lower)
        data_score = sum(1 for keyword in data_keywords if keyword in question_lower)
        
        # Quyết định dựa trên score
        if image_score > data_score:
            return "image"
        elif data_score > image_score:
            return "data"
        else:
            # Nếu không rõ ràng, ưu tiên data
            return "data"

class QASystem:
    def __init__(self):
        print("🚀 Khởi tạo QA System...")
        
        self.llm = Qwen2VLLLM()
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.daily_report_data = None
        self.raw_data = None
        self.vector_store = None
        self.current_image_path = "data/test1.png"  # Default image path
        
        self.load_initial_data()
        print("✅ QA System sẵn sàng!")
    
    def load_initial_data(self):
        """Tải tất cả dữ liệu có sẵn khi khởi chạy"""
        try:
            # Load daily report
            daily_report_path = "data/daily_report_20250628.csv"
            if os.path.exists(daily_report_path):
                self.daily_report_data = pd.read_csv(daily_report_path)
                print(f"✅ Đã tải daily report: {len(self.daily_report_data)} dòng")
            else:
                print("⚠️ Không tìm thấy daily report")
            
            # Load raw data (demotest1.csv)
            raw_data_path = "data/demotest1.csv"
            if os.path.exists(raw_data_path):
                self.raw_data = pd.read_csv(raw_data_path)
                print(f"✅ Đã tải raw data: {len(self.raw_data)} dòng")
            else:
                print("⚠️ Không tìm thấy raw data")
            
            # Verify image path
            if os.path.exists(self.current_image_path):
                print("✅ Đã tìm thấy ảnh dashboard")
            else:
                print("⚠️ Không tìm thấy ảnh dashboard")
                # Fallback to old path if exists
                if os.path.exists("test1.png"):
                    self.current_image_path = "test1.png"
                    print("✅ Sử dụng ảnh mặc định test1.png")
            
            # Build vector store if any data is loaded
            if self.daily_report_data is not None or self.raw_data is not None:
                self.build_vector_store()
                
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu ban đầu: {e}")
    
    def get_data_summary(self) -> str:
        """Tạo summary về dữ liệu hiện có"""
        summary_parts = []
        
        if self.daily_report_data is not None:
            total_volume = self.daily_report_data['volume'].sum() if 'volume' in self.daily_report_data.columns else 0
            total_bu = len(self.daily_report_data)
            summary_parts.append(f"📊 **Daily Report**: {total_bu} BU, tổng volume: {total_volume:,.0f}")
        
        if self.raw_data is not None:
            total_records = len(self.raw_data)
            if 'vol_sellout_kat' in self.raw_data.columns:
                total_sellout = self.raw_data['vol_sellout_kat'].sum()
                summary_parts.append(f"📈 **Raw Data**: {total_records:,} records, tổng sellout: {total_sellout:,.0f}")
            else:
                summary_parts.append(f"📈 **Raw Data**: {total_records:,} records")
        
        if os.path.exists(self.current_image_path):
            summary_parts.append(f"🖼️ **Dashboard**: {os.path.basename(self.current_image_path)}")
        
        if not summary_parts:
            return "❌ Chưa có dữ liệu nào được tải"
        
        return "\n".join(summary_parts)
    
    def process_csv_file(self, file_path: str, is_raw_data: bool = False):
        try:
            df = pd.read_csv(file_path)
            if is_raw_data:
                self.raw_data = df
            else:
                self.daily_report_data = df
            
            self.build_vector_store()
            return f"✅ Đã tải CSV thành công ({len(df)} dòng)"
            
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"
    
    def process_image(self, image_path: str):
        """Xử lý ảnh được upload"""
        try:
            if image_path and os.path.exists(image_path):
                self.current_image_path = image_path
                return f"✅ Đã tải ảnh thành công", image_path
            else:
                return "❌ Không tìm thấy ảnh", None
        except Exception as e:
            return f"❌ Lỗi: {str(e)}", None
    
    def build_vector_store(self):
        try:
            documents = []
            
            if self.daily_report_data is not None:
                daily_text = f"Dữ liệu Daily Report:\nCác cột: {', '.join(self.daily_report_data.columns.tolist())}\n"
                
                for _, row in self.daily_report_data.iterrows():
                    bu_text = f"Đơn vị {row['BU']} (Mã: {row['BU_CODE']}): "
                    bu_text += f"Khối lượng: {row['volume']}, "
                    bu_text += f"Target MK: {row['target_mk_volume']}, Target SK: {row['target_sk_volume']}, "
                    bu_text += f"Volume MK: {row['volume_MK']}, Volume SK: {row['volume_SK']}, "
                    bu_text += f"% MK: {row['%_mk_done']}%, % SK: {row['%_sk_done']}%"
                    
                    documents.append(Document(
                        page_content=bu_text,
                        metadata={"source": f"daily_{row['BU_CODE']}", "type": "daily"}
                    ))
                
                documents.append(Document(
                    page_content=daily_text,
                    metadata={"source": "daily_summary", "type": "summary"}
                ))
            
            if self.raw_data is not None:
                raw_text = f"Dữ liệu chi tiết:\nCác cột: {', '.join(self.raw_data.columns.tolist())}\n"
                
                if 'bu_name' in self.raw_data.columns:
                    bu_summary = self.raw_data.groupby('bu_name').agg({
                        'vol_sellout_kat': 'sum',
                        'rev_sellout_kat_vat': 'sum'
                    }).reset_index()
                    
                    for _, row in bu_summary.iterrows():
                        bu_detail = f"Đơn vị {row['bu_name']}: "
                        bu_detail += f"Volume: {row['vol_sellout_kat']}, "
                        bu_detail += f"Doanh thu: {row['rev_sellout_kat_vat']}"
                        
                        documents.append(Document(
                            page_content=bu_detail,
                            metadata={"source": f"raw_{row['bu_name']}", "type": "raw"}
                        ))
                
                documents.append(Document(
                    page_content=raw_text,
                    metadata={"source": "raw_summary", "type": "summary"}
                ))
            
            if documents:
                split_documents = self.text_splitter.split_documents(documents)
                self.vector_store = FAISS.from_documents(split_documents, self.embeddings)
                print(f"✅ Vector store: {len(split_documents)} chunks")
                
        except Exception as e:
            print(f"Lỗi vector store: {e}")
    
    def answer_question(self, question: str) -> str:
        """Trả lời câu hỏi với logic phân loại thông minh"""
        try:
            # Phân loại câu hỏi
            question_type = self.llm.classify_question(question)
            
            has_data = (self.daily_report_data is not None or self.raw_data is not None)
            has_image = os.path.exists(self.current_image_path)
            
            if not has_data and not has_image:
                return "❓ Không có dữ liệu để trả lời. Vui lòng tải dữ liệu lên."
            
            # Xử lý theo loại câu hỏi
            if question_type == "image" and has_image:
                # Câu hỏi về ảnh - chỉ phân tích ảnh
                try:
                    image_analysis = self.llm.analyze_image_with_text(self.current_image_path, question)
                    return f"📊 **Phân tích dashboard:**\n{image_analysis}"
                except Exception as e:
                    return f"⚠️ Lỗi phân tích ảnh: {str(e)}"
            
            elif question_type == "data" and has_data and self.vector_store:
                # Câu hỏi về dữ liệu - chỉ phân tích dữ liệu
                try:
                    retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
                    relevant_docs = retriever.get_relevant_documents(question)
                    
                    if relevant_docs:
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        enhanced_prompt = f"""
                        Bạn là một chuyên gia phân tích dữ liệu kinh doanh. 
                        Dựa trên dữ liệu sau, hãy trả lời câu hỏi một cách chi tiết và chính xác:
                        
                        Dữ liệu: {context}
                        
                        Câu hỏi: {question}
                        
                        Yêu cầu:
                        - Trả lời bằng tiếng Việt
                        - Đưa ra số liệu cụ thể
                        - Phân tích ngắn gọn và rõ ràng
                        - Nếu có so sánh, hãy làm rõ
                        """
                        
                        data_analysis = self.llm._call(enhanced_prompt)
                        return f"📈 **Phân tích dữ liệu:**\n{data_analysis}"
                    else:
                        return "❓ Không tìm thấy dữ liệu phù hợp với câu hỏi."
                        
                except Exception as e:
                    return f"⚠️ Lỗi phân tích dữ liệu: {str(e)}"
            
            else:
                # Fallback - thử cả hai nếu không phân loại được rõ
                response_parts = []
                
                if has_image:
                    try:
                        image_analysis = self.llm.analyze_image_with_text(self.current_image_path, question)
                        response_parts.append(f"📊 **Phân tích dashboard:**\n{image_analysis}")
                    except Exception as e:
                        response_parts.append(f"⚠️ Lỗi phân tích ảnh: {str(e)}")
                
                if has_data and self.vector_store:
                    try:
                        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                        relevant_docs = retriever.get_relevant_documents(question)
                        
                        if relevant_docs:
                            context = "\n\n".join([doc.page_content for doc in relevant_docs])
                            
                            enhanced_prompt = f"""
                            Dựa trên dữ liệu sau, trả lời câu hỏi bằng tiếng Việt:
                            
                            Dữ liệu: {context}
                            
                            Câu hỏi: {question}
                            
                            Trả lời ngắn gọn, rõ ràng với số liệu cụ thể.
                            """
                            
                            data_analysis = self.llm._call(enhanced_prompt)
                            response_parts.append(f"📈 **Phân tích dữ liệu:**\n{data_analysis}")
                        
                    except Exception as e:
                        response_parts.append(f"⚠️ Lỗi phân tích dữ liệu: {str(e)}")
                
                if response_parts:
                    return "\n\n".join(response_parts)
                else:
                    return "❓ Không tìm thấy thông tin phù hợp."
            
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"

def create_interface():
    qa_system = QASystem()
    
    def process_csv(file, is_raw):
        if file:
            result = qa_system.process_csv_file(file.name, is_raw)
            # Cập nhật summary sau khi tải dữ liệu
            new_summary = qa_system.get_data_summary()
            return result, new_summary
        return "Chưa chọn file", qa_system.get_data_summary()
    
    def process_image_upload(file):
        if file:
            status, image_path = qa_system.process_image(file.name)
            new_summary = qa_system.get_data_summary()
            return status, image_path, new_summary
        return "Chưa chọn ảnh", None, qa_system.get_data_summary()
    
    def get_default_image():
        if os.path.exists("data/test1.png"):
            return "data/test1.png"
        elif os.path.exists("test1.png"):
            return "test1.png"
        return None
    
    def get_initial_status():
        """Lấy trạng thái ban đầu của dữ liệu đã load"""
        status = []
        
        # Check daily report
        if qa_system.daily_report_data is not None:
            status.append(f"✅ Daily Report: {len(qa_system.daily_report_data)} dòng")
        else:
            status.append("❌ Daily Report: Chưa tải")
        
        # Check raw data
        if qa_system.raw_data is not None:
            status.append(f"✅ Raw Data: {len(qa_system.raw_data)} dòng")
        else:
            status.append("❌ Raw Data: Chưa tải")
        
        # Check image
        if os.path.exists(qa_system.current_image_path):
            status.append(f"✅ Ảnh: {qa_system.current_image_path}")
        else:
            status.append("❌ Ảnh: Không tìm thấy")
        
        return "\n".join(status)
    
    def chat_fn(message, history):
        response = qa_system.answer_question(message)
        history.append([message, response])
        return history, ""
    
    with gr.Blocks(title="QA System", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 QA System với Phân tích Thông minh")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image display với khả năng upload
                image_display = gr.Image(
                    value=get_default_image(),
                    label="📊 Dashboard", 
                    height=400,
                    interactive=False
                )
                
                # Upload ảnh mới
                with gr.Accordion("📷 Thay đổi ảnh", open=False):
                    with gr.Row():
                        image_upload = gr.File(
                            label="Tải ảnh mới", 
                            file_types=[".png", ".jpg", ".jpeg"],
                            scale=3
                        )
                        upload_img_btn = gr.Button("📷 Tải ảnh", scale=1)
                    
                    img_status = gr.Textbox(
                        label="Trạng thái ảnh", 
                        interactive=False,
                        value=f"Sử dụng ảnh: {qa_system.current_image_path}" if os.path.exists(qa_system.current_image_path) else "Chưa có ảnh"
                    )
                
                with gr.Accordion("📊 Tải dữ liệu mới", open=False):
                    # Hiển thị trạng thái dữ liệu đã load
                    gr.Textbox(
                        label="📋 Dữ liệu hiện tại",
                        value=get_initial_status(),
                        interactive=False,
                        lines=4
                    )
                    
                    gr.Markdown("---")
                    
                    with gr.Accordion("📈 Daily Report", open=False):
                        daily_upload = gr.File(label="Chọn file CSV", file_types=[".csv"])
                        daily_btn = gr.Button("📤 Tải Daily Report")
                        daily_status = gr.Textbox(label="Kết quả", interactive=False)
                    
                    with gr.Accordion("📊 Raw Data", open=False):
                        raw_upload = gr.File(label="Chọn file CSV", file_types=[".csv"])
                        raw_btn = gr.Button("📤 Tải Raw Data")
                        raw_status = gr.Textbox(label="Kết quả", interactive=False)
            
            with gr.Column(scale=1):
                # Component Summary mới
                with gr.Accordion("📋 Tổng quan dữ liệu", open=True):
                    data_summary = gr.Markdown(
                        value=qa_system.get_data_summary(),
                        label="Tóm tắt dữ liệu"
                    )
                
                # Chatbot
                chatbot = gr.Chatbot(height=350, label="💬 Trò chuyện")
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Nhập câu hỏi (hệ thống sẽ tự động phân loại câu hỏi về ảnh hay dữ liệu)...", 
                        container=False, 
                        scale=5
                    )
                    send_btn = gr.Button("📤 Gửi", scale=1)
                
                clear = gr.Button("🗑️ Xóa lịch sử")
                
                # Thêm hướng dẫn
                gr.Markdown("""
                **💡 Hướng dẫn sử dụng:**
                - Câu hỏi về **ảnh/dashboard**: "Ảnh này hiển thị gì?", "Màu sắc trong biểu đồ", "Giao diện như thế nào?"
                - Câu hỏi về **dữ liệu**: "Tổng volume là bao nhiêu?", "BU nào có doanh thu cao nhất?", "So sánh target và actual"
                """)
        
        # Event handlers
        upload_img_btn.click(
            fn=process_image_upload,
            inputs=image_upload,
            outputs=[img_status, image_display, data_summary]
        )
        
        daily_btn.click(
            fn=lambda f: process_csv(f, False),
            inputs=daily_upload,
            outputs=[daily_status, data_summary]
        )
        
        raw_btn.click(
            fn=lambda f: process_csv(f, True),
            inputs=raw_upload,
            outputs=[raw_status, data_summary]
        )
        
        msg.submit(chat_fn, [msg, chatbot], [chatbot, msg])
        send_btn.click(chat_fn, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: [], outputs=chatbot)
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )