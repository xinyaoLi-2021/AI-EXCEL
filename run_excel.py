import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import json
import os
from datetime import datetime
import subprocess
import threading
import ast
import re
#from ollama import Client 
from openai import OpenAI
from openai.types.chat.chat_completion import Choice

class CodeExecutor:
    """代码执行器类，负责解析和执行Python代码"""
        
    def parse_code_from_string(self, input_string):
        """
        从字符串中解析可执行代码
        """
        # 匹配三个反引号的代码块
        triple_backtick_pattern = r"```(?:\w*\s*)?(.*?)```"
        match = re.search(triple_backtick_pattern, input_string, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(2).strip()
        
        # 匹配单个反引号的代码块
        single_backtick_pattern = r"`(.*?)`"
        match = re.search(single_backtick_pattern, input_string, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 如果没有找到代码块，返回原字符串
        return input_string.strip()
    

    def object_to_string(self, obj, command=''):
        """
        将对象转换为字符串表示
        """
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, pd.DataFrame):
            if len(obj) == 0:
                return '空DataFrame'
            return obj.to_string(index=False, max_rows=20)
        elif isinstance(obj, pd.Series):
            return obj.to_string()
        elif command == 'df.columns':
            cols = obj.tolist()
            if len(cols) > 20:
                return str(cols[:10])[:-1] + ', ..., ' + str(cols[-10:])[1:]
        return str(obj)
    

    def python_repl(self, code, custom_locals=None, custom_globals=None, memory=None):
        """
        执行Python代码
        """
        if custom_locals is None:
            custom_locals = {}
        if custom_globals is None:
            custom_globals = {}
        if memory is None:
            memory = {}

        output = ""
        observation = "执行完成"
        

        try:
            
            # 解析和执行代码
            tree = ast.parse(code)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            
            # 执行除最后一行外的所有代码
            exec(ast.unparse(module), custom_globals, custom_locals)
            
            # 处理最后一行
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)
            
            
            if module_end_str.startswith('print('):
                module_end_str = module_end_str.strip()[6:-1]
           

            try:
                ret = eval(module_end_str, custom_globals, custom_locals)
                if ret is not None:
                    output = self.object_to_string( ret, module_end_str)
                    print(f"=== 计算结果: {output} ")
                    
                else:
                    output = "None"
                    print("=== 计算结果为 None")

            except Exception as e:
                
                # 构建包含错误信息的output
                import traceback
                traceback_str = traceback.format_exc()
                error_msg = f"eval失败: {str(e)}"

                output = f"错误: {error_msg}\n详细追踪:\n{traceback_str}"
                observation = "执行出错"
                
                print(f"=== {error_msg}")

            memory.update(custom_locals)
            return observation, output
        

        except Exception as e:
            print(f"=== 执行出错: {e} ===")
            import traceback
            traceback_str = traceback.format_exc()
            return "解析错误", f"错误: {e}\n{traceback_str}"
    



class ReactSolver:

    def __init__(self, code_executor: CodeExecutor, csv_tool):
        
        #self.ollama_client = Client()
        self.client = OpenAI(
            api_key = "sk-a1wsGcMV9SqSYFLYUPbNspttFJpGUUelEF5MlaT00M4wFicX",
            base_url = "https://api.moonshot.cn/v1",
        )
        self.code_executor = code_executor
        self.csv_tool = csv_tool
        self.model_name = "kimi-k2-turbo-preview"
        self.temperature = 0.8
        self.max_depth = 20
        # ReAct求解状态
        self.is_solving = False
        self.current_iteration = 0
        self.current_df = None
        self.current_query = ""
        self.memory = {}
        self.solution = ""
        self.current_prompt = ""
        self.current_code = ""
        self.waiting_for_execution = False


    def clean_response(self, text):
        """清理响应，确保只有Thought 和Action"""
        lines = text.split('\n')
        
        # 提取Thought行
        thought_lines = [line for line in lines if line.startswith('Thought:')]
        thought = thought_lines[0] if thought_lines else "Thought: 分析问题"
        
        # 提取Action行  
        action_lines = [line for line in lines if line.startswith('Action:')]
        action = action_lines[0] if action_lines else "Action: print(df.head())"
        
        # 返回标准格式
        return f"{thought}\n{action}"
    

    def response(self, prompt, mode) -> str:
        
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=2000  
        )
            if mode == 'easy':
                response_raw_text = response.choices[0].message.content.strip()
                response_text = self.clean_response(response_raw_text)
            else:
                response_text = response.choices[0].message.content.strip()
            
            return response_text

        except Exception as e:
            return f"Error: {str(e)}"
             
        

    def get_prompt(self, full_query_text, mode):
        #print('//'*10,full_query_text, mode )

        if "问题：" in full_query_text:
            parts = full_query_text.split("问题：", 1)
            context_info = parts[0].strip()  # 上下文信息
            user_query = parts[1].strip()    # 用户查询

            context_info_clean = context_info.replace('#', '').strip()
        else:
            context_info_clean = ""
            user_query = full_query_text.strip()
        #print('$'*10,context_info_clean, user_query)

        # 获取实际路径值
        input_folder = self.csv_tool.selected_folder_4_input
        #print('//'*10, input_folder)

        output_folder = self.csv_tool.selected_folder_4_output if hasattr(self.csv_tool, 'selected_folder_4_output') and self.csv_tool.selected_folder_4_output else './output'
        #selected_file = self.csv_tool.file_combobox.get()
        #file_path = os.path.join(self.csv_tool.selected_folder_4_output, selected_file)


        # 构建prompt for mode='easy
        base_prompt = f'''
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. The DataFrame contains the following Information::{context_info_clean}
Your task is to use `python_repl_ast` to answer the question: {user_query}

Tool description:
- `python_repl_ast`: A Python interactive shell. Use this to execute python commands. Input should be a valid single line python command.

**STRICT FORMAT REQUIREMENTS:**
You MUST respond with EXACTLY TWO lines:
Thought: you should always think about what to do
Action: the single line Python command to execute

Do NOT include any other text, explanations.

Begin!
'''   
        
        # only construct prompt for pipeline
        pipeline_prompt = f'''
You are a pandas dataframe analysis expert performing a multi-step CSV analysis.

### Background
- The original input CSV from: {input_folder}.
- The DataFrame information: {context_info_clean}.
- User query: {user_query}.

### Task & Constraints
- Break down the analysis into logical steps
- Only use operations that: add/delete columns OR delete cells
- Each step should generate a new CSV that serves as input for the next step
- Focus on core concepts and key data processing steps
- Use English brackets: [Step X]

### Step Format (STRICTLY FOLLOW)
[Step X]
Observation: <What's already known from previous steps>
Thought: <Reasoning about this step's action>

'''
        

        pipeline_prompt_each_step = f'''
You are a Python data analysis expert. Write a python script to answer the Thought in the following task:{full_query_text}.

### Important: Output Locations
- **Output folder**: {output_folder}

### Rule:
First read last Steps Result, use pd.columns()
ALWAYS save to: {output_folder}

## Requirements:
1. Output must be valid JSON with these keys: "code", "out_script_name", "command"
2. Use ASCII only, no Chinese characters
3. Output files should have descriptive names
4. Include necessary imports and error handling
5. Save all output CSV files to: {output_folder} 
6. Generate descriptive filenames
'''

   
        
        if mode == 'easy': #简单React
            prompt = base_prompt
        elif mode == 'complex': #流程化
            prompt = pipeline_prompt
        elif mode == 'each_step':
            prompt = pipeline_prompt_each_step
        else:
            prompt = base_prompt
        return prompt
            
    

    def start_solver_loop(self, df: pd.DataFrame, query: str) -> str:
        if self.is_solving:
            self.csv_tool.output_display("正在执行中，请等待...\n")
            return
        
        self.is_solving = True
        self.current_iteration = 0
        self.current_df = df.copy()
        self.current_prompt = query
        memory = {}
        self.solution = ''
        self.current_prompt = self.get_prompt(query,'easy')
        self.csv_tool.output_display(f"\n=== 开始处理Query ===\nQuery: {query}\n")
        self.csv_tool.output_display(f"初始提示:\n{self.current_prompt}\n")
        self.solver_iteration()

    
    # (Solver loop follows the ReAct framework: https://github.com/ysymyth/ReAct.)
    def solver_iteration(self) -> str:
        print('self.is_solving:' , self.is_solving)
        if not self.is_solving or self.current_iteration >= self.max_depth:
            self.finish_solver()
            return
        
        self.current_iteration += 1  
        self.csv_tool.output_display(f"\n--- 迭代 {self.current_iteration} ---\n")

        self.solution += 'Thought: ' # always start with thoughts 
        prompt = self.current_prompt + self.solution
        text = self.response(prompt,'easy').strip()  #给出答案


        if 'Action:' not in text:
            self.csv_tool.output_display(f"错误: Ollama响应中没有Action, {text}\n")
            self.finish_solver()
            return
        else:
            # execute the code, we need to pass the dataframe, and pandas as pd, numpy as np to the locals
            code = self.code_executor.parse_code_from_string(text.split('Action:')[-1].strip())   
            self.current_code = code  

            self.csv_tool.output_display(f"Ollama响应:\n{text}\n")

            # 将代码插入到GUI的代码区域
            self.csv_tool.code_text.delete("1.0", tk.END)
            self.csv_tool.code_text.insert("1.0", code)

            # 设置等待执行状态
            self.waiting_for_execution = True
            self.csv_tool.output_display("请点击'执行Action'按钮来执行代码...\n")

            # 更新解决方案记录
            self.solution = text + '\n' + '等待代码执行...\n'
            

    def execute_current_code(self):
        if not self.waiting_for_execution or not self.is_solving:
            return
        
        code = self.current_code
        self.csv_tool.output_display(f"\n执行代码: {code}\n")

        try:
                
            observation, output = self.code_executor.python_repl(
                code, 
                custom_locals={'df': self.current_df.copy(), 'pd': pd}, 
                custom_globals=globals(), 
                memory=self.memory)               
            
            if output and output != "None":
                self.csv_tool.output_display(f"输出内容:\n{output}\n")

            # 更新解决方案记录
            self.solution = self.solution.replace('等待代码执行...', f'执行结果:\n{output}\n')
            self.waiting_for_execution = False
                
            self.csv_tool.root.after(10, self.solver_iteration)

        except Exception as e:
            self.csv_tool.output_display(f"执行出错: {str(e)}\n")
            self.finish_solver()
                
                       
    def finish_solver(self):
        """结束求解过程"""
        self.is_solving = False
        self.waiting_for_execution = False
        self.csv_tool.output_display(f"总共迭代次数: {self.current_iteration}\n")
        
       
import tkinter as tk
from tkinter import ttk

class ArrowSeparator(tk.Frame):  
    """箭头分隔符"""
    def __init__(self, parent, **kwargs):
        # 获取父容器的背景色
        try:
            bg_color = parent.cget('background')
        except:
            # 如果父容器是ttk部件，使用默认背景色
            temp = tk.Frame(parent)
            bg_color = temp.cget('background')
            temp.destroy()
        
        # 初始化Frame
        super().__init__(parent, **kwargs)
        
        # 设置Frame背景色
        self.configure(bg=bg_color)
        
        # 创建Canvas
        self.canvas = tk.Canvas(
            self, 
            width=40, 
            height=20, 
            highlightthickness=0,
            bg=bg_color,
            bd=0
        )
        self.canvas.pack()
        
        # 绘制箭头
        self.draw_arrow()
    
    def draw_arrow(self):
        """绘制箭头"""
        self.canvas.create_line(
            10, 10,
            30, 10,
            width=2,
            arrow=tk.LAST,
            fill='#555555'
        )

from datetime import datetime
class CSVQueryTool:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Tool")
        self.root.geometry("1000x700")
        
        # 选择的文件夹路径
        self.selected_folder_4_input = None
        self.selected_folder_4_output = None
        
        # 当前加载的CSV文件
        self.current_df = None
        # 内存变量
        self.memory = {}

        # 初始化执行器和求解器 CLASS
        self.code_executor = CodeExecutor()
        self.react_solver = ReactSolver(self.code_executor, self)

        self.setup_ui()

        #存储
        self.prompt_4_save = ''
        self.memory_4_query_pipeline = ''


    

    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板
        left_frame = ttk.LabelFrame(main_frame, text="输入", width=500)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # 右侧面板
        right_frame = ttk.LabelFrame(main_frame, text="输出", width=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_frame.pack_propagate(False)
        
        # 设置左侧面板
        self.setup_left_panel(left_frame)
        
        # 设置右侧面板
        self.setup_right_panel(right_frame)
    
    def setup_left_panel(self, parent):
        # 文件夹选择区域
        folder_frame = ttk.Frame(parent)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(folder_frame, text="工作目录:").pack(side=tk.LEFT)
        self.folder_label_left = ttk.Label(folder_frame, text="未选择", foreground="red")
        self.folder_label_left.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(folder_frame, text="选择文件夹", 
                  command=self.select_folder_4_input).pack(side=tk.LEFT)
        
        # CSV文件选择区域 ----------------
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="CSV文件:").pack(side=tk.LEFT)
        self.file_combobox = ttk.Combobox(file_frame, state="readonly", width=30)
        self.file_combobox.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(file_frame, text="刷新", 
                  command=self.refresh_files).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(file_frame, text="加载", 
                  command=self.load_selected_file).pack(side=tk.LEFT)
        
        
        #----------------------------创建包含Query标签/说明的框架/模式选择框 -------------------------------------------
        query_label_frame = ttk.Frame(parent)
        query_label_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(query_label_frame, text="Query:").pack(side=tk.LEFT)
        ttk.Button(query_label_frame, text="Execute Query",
                  command=self.execute_query).pack(side=tk.LEFT, padx=(10, 0))

        # help(middle)
        """
        mode_help = ttk.Label(
            query_label_frame, 
            text="简单模式 | 复杂模式(多步骤分析任务)",
            foreground="gray",
            font=("Arial", 8)
        )
        mode_help.pack(side=tk.LEFT, padx=(10, 0))

        # mode(right)
        mode_frame = ttk.Frame(query_label_frame)
        mode_frame.pack(side=tk.RIGHT)
        ttk.Label(mode_frame, text="模式:").pack(side=tk.LEFT)
        self.query_mode = ttk.Combobox(
            mode_frame, 
            values=["简单模式", "复杂模式"],
            state="readonly",
            width=15)
        self.query_mode.set("简单")
        self.query_mode.pack(side=tk.LEFT, padx=(5, 0))
        """

        # up/down query box 
        query_main_frame = ttk.Frame(parent)
        query_main_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        #self.query_input = scrolledtext.ScrolledText(parent, height=20)
        #self.query_input.pack(fill=tk.BOTH, expand=True)
        # 上半部分：用户输入区（白色背景）
        self.query_input = scrolledtext.ScrolledText(query_main_frame, height=20)
        self.query_input.pack(fill=tk.BOTH, expand=True)
        # 下半部分：正在执行的查询显示区（灰色背景）
        self.running_query_display = scrolledtext.ScrolledText(
            query_main_frame, 
            height=8,
            background="#f5f5f5",  # 浅灰色背景
            foreground="#333333"   # 深灰色文字
        )
        self.running_query_display.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        # 可选：添加分割线
        separator = ttk.Separator(query_main_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=5)
        

        # 按钮区域 ------箭头--------------
        query_button_frame = ttk.Frame(parent)
        query_button_frame.pack(fill=tk.X, pady=(0, 10))
        
        #ttk.Button(query_button_frame, text="Execute Query",
                  #command=self.execute_query).pack(side=tk.RIGHT, padx=(0, 10))
        
        ttk.Button(query_button_frame, text="Query Pipeline", 
                  command=self.prepare_4_query_pipeline).pack(side=tk.LEFT, padx=(0, 2))    
        
        # 1. Arror
        self.arrow_separator1 = ArrowSeparator(query_button_frame, width=20)
        self.arrow_separator1.pack(side=tk.LEFT, padx=2)

        # "Next Step" Button
        self.single_step_btn = ttk.Button(
            query_button_frame, 
            text="Execute",
            command=self.execute_each_step,
            state=tk.DISABLED  # 初始禁用
        )
        self.single_step_btn.pack(side=tk.LEFT, padx=(0, 2))

        """
        # 2. Arror
        self.arrow_separator2 = ArrowSeparator(query_button_frame, width=20)
        self.arrow_separator2.pack(side=tk.LEFT, padx=2)

        # "Execute" Button
        
        self.execute_btn = ttk.Button(
            query_button_frame, 
            text="Execute",
            command=self.execute_single_step,
            state=tk.DISABLED  # 初始禁用
        )
        self.execute_btn.pack(side=tk.LEFT)
        """
        
        # -----------------------------Action代码区域------------------------------------
        ttk.Label(parent, text="Action代码/Command:").pack(anchor=tk.W, pady=(10, 5))
        self.code_text = scrolledtext.ScrolledText(parent, height=2)
        self.code_text.pack(fill=tk.BOTH, expand=True)

        # 执行Action按钮区域
        action_button_frame = ttk.Frame(parent)
        action_button_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(action_button_frame, text="执行Action/Command", 
                  command=self.execute_action).pack(side=tk.RIGHT)
        
        
        self.status_label = ttk.Label(parent, text="就绪", foreground="green")
        self.status_label.pack(anchor=tk.W, pady=(5, 0))


    def setup_right_panel(self, parent):
        # 输出文件选择区域
        folder_frame = ttk.Frame(parent)
        folder_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(folder_frame, text="输出文件夹:").pack(side=tk.LEFT)
        self.folder_label_right = ttk.Label(folder_frame, text="未选择", foreground="red")
        self.folder_label_right.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Button(folder_frame, text="选择文件夹", 
                  command=self.select_folder_4_output).pack(side=tk.LEFT)
        
        # 输出区域
        ttk.Label(parent, text="执行输出:").pack(anchor=tk.W, pady=(0, 5))
        self.output_text = scrolledtext.ScrolledText(parent, height=25)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        

        # 底部按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="清除输出", 
                  command=self.clear_output).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="保存结果", 
                  command=self.save_result).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="停止求解", 
                  command=self.stop_solving).pack(side=tk.RIGHT)
        
    def select_folder_4_output(self):
        """选择输出code和csv文件夹"""
        folder_path = filedialog.askdirectory(title="选择输出文件夹")
        if folder_path:
            self.selected_folder_4_output = os.path.normpath(folder_path)
            os.chdir(self.selected_folder_4_output)
            self.folder_label_right.config(text=folder_path, foreground="green")
        



    def select_folder_4_input(self):
        """选择工作文件夹"""
        folder_path = filedialog.askdirectory(title="选择工作文件夹")
        if folder_path:
            self.selected_folder_4_input = os.path.normpath(folder_path)
            os.chdir(self.selected_folder_4_input)
            self.folder_label_left.config(text=os.path.basename(folder_path), foreground="green")
            self.refresh_files()
            
            # 显示选择结果
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output = f"=== 文件夹选择 ({timestamp}) ===\n"
            output += f"工作目录: {folder_path}\n"
            output += "=" * 50 + "\n\n"
            self.output_display(output)
    

    def refresh_files(self):
        """刷新文件列表"""
        if not self.selected_folder_4_input:
            messagebox.showwarning("提示", "请先选择工作文件夹")
            return
        
        files = []
        for file in os.listdir(self.selected_folder_4_input):
            if file.lower().endswith(('.csv', '.xslx', '.txt')):
                files.append(file)
        #下拉
        self.file_combobox['values'] = files
        if files:
            self.file_combobox.set(files[0])
    


    def load_selected_file(self):
        """加载选中的文件, 并放入query"""
        if not self.selected_folder_4_input:
            messagebox.showwarning("提示", "请先选择工作文件夹")
            return
        
        selected_file = self.file_combobox.get() # 获取左侧combobox的选择
        if not selected_file:
            messagebox.showwarning("提示", "请选择文件")
            return
        
        try:
            file_path = os.path.join(self.selected_folder_4_input, selected_file)
            file_extension = os.path.splitext(selected_file)[1].lower()
        
            if file_extension in ['.csv', '.txt']:
                self.current_df = pd.read_csv(file_path, sep=None, engine='python')
            elif file_extension in ['.xls', '.xlsx']:
                self.current_df = pd.read_excel(file_path)
            else:
                messagebox.showwarning("不支持", f"不支持的文件格式: {file_extension}")
                return
            
            if self.current_df is None:
                return
        
            # 输出到结果区域
            #self.output_text.delete("1.0", tk.END)
            info = f"文件: {selected_file}\n行数: {len(self.current_df)} 行\n列数: {len(self.current_df.columns)} 列\n\n"
            info += "前5行:\n" + self.current_df.head().to_string()
            self.output_text.insert("1.0", info)
            self.output_display(f"已加载: {selected_file} ({len(self.current_df)}行×{len(self.current_df.columns)}列)\n")

            # 同时输出<首行>到 query_input
            self.query_input.delete("1.0", tk.END)
            info_2 = f"# File: {selected_file}\n"
            info_2 += f"# Columns: {list(self.current_df.columns)}\n"
            info_2 += "#" * 40 + "\n\n"
            info_2 += "问题：\n"
            self.query_input.insert("1.0", info_2)

            # 设置光标到"用户问题："后面
            self.query_input.mark_set("insert", "end-1c lineend")
            self.query_input.focus()

        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败: {str(e)}")
    

#######################################################################################################################
    def prepare_4_query_pipeline(self):
        """Query流程化"""
        if self.current_df is None:
            messagebox.showwarning("提示", "请先加载CSV文件")
            return
        
        query = self.query_input.get("1.0", tk.END).strip()
        #print('!!'*10,query) 
        if not query:
            messagebox.showwarning("提示", "请输入Query")
            return
        
        self.prompt = self.react_solver.get_prompt(query, 'complex')
        print('--'*10) 
        print('Prompt:',self.prompt) 

        # save prompt
        self.prompt_4_save += f"\n{self.prompt}\n"

        self.query_pipeline(self.prompt)


    def query_pipeline(self, prompt):

        """Query流程化"""
        error_exist = True
        retry_count = 0
        while error_exist and retry_count < 3:  # 最多重试3次
            text = self.react_solver.response(prompt,'complex').strip()  #给出答案
            if 'error' in text.lower():
                error_exist = True
                retry_count += 1
                print(f"检测到error，第{retry_count}次重试...")
            else:
                error_exist = False

        print('--'*10) 
        print('text:',text) 

        self.query_input.delete("1.0", tk.END)
        self.query_input.insert("1.0", text)


        # 解析text为步骤列表
        steps = self.parse_steps_from_text(text)

        # 保存步骤到实例变量
        self.pipeline_steps = steps
        self.current_step_index = 0

        # 清空显示区域 不对
        self.running_query_display.delete("1.0", tk.END)
        step_obj = self.pipeline_steps[self.current_step_index]

        display_text = ""
        # 如果有步骤，在显示区域添加标题
        if steps:
            step_number = step_obj.get('number', self.current_step_index + 1)
            display_text += f"Running Step {step_number}:\n"
            display_text += f"{'-'*50}\n"
            display_text += f"{step_obj['content']}\n"

            # 添加到显示区域
            self.running_query_display.insert(tk.END, display_text)
            
        # Query流程化 -> 启用单步执行按钮 
        self.single_step_btn.config(state=tk.NORMAL)
        

        return
    
    def parse_steps_from_text(self, text):
        """从文本中解析出步骤"""
        import re
        
        # 模式：匹配 [Step 数字] 和直到下一个 [Step]或文本结束的内容
        pattern = r'\[Step\s*(\d+)\]([\s\S]*?)(?=\[Step\s*\d+\]|$)'


        try:
            matches = re.findall(pattern, text)
        except re.error as e:
            print(f"正则表达式错误: {e}")
            print(f"使用的模式: {pattern}")
            return []
        
        steps = []
        for step_num, content in matches:
            content = content.strip()
            lines = content.split('\n')

            cleaned_lines = []
            for line in lines:
                line = line.strip()

                # 1. 检查并去掉包含 "Step" 的行
                if '[Step' in line:
                    continue  

                # 2. 去掉以 '-' 开头的行
                if line.startswith('-'):
                    line = line[1:].strip()

                # 3. 去掉以 '---' 结尾的行
                if line.endswith('---'):
                    line = line[:-3].strip()
                
                # 4. 只保留非空行
                if line:
                    cleaned_lines.append(line)
            
            content = '\n'.join(cleaned_lines)

            steps.append({
            'number': int(step_num),
            'content': content
        })
            
        return steps

    
    #### DODODODOODODODODOdododo-----------------------------------------------~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def locate_json_string_body_from_string(self, content: str):
    
        #Locate the JSON string body from a string
        try:
            if content.startswith("{") and content.endswith("}"):
                return json.loads(content)
            
            maybe_json_str = re.search(r"json\s*[\[|\{].*[\]|\}]\s", content, re.DOTALL)
            #print(maybe_json_str)
            if maybe_json_str is not None:
                maybe_json_str = maybe_json_str.group(0)
                #maybe_json_str = maybe_json_str.replace("\\n", "")
                #maybe_json_str = maybe_json_str.replace("\n", "")
                #maybe_json_str = maybe_json_str.replace("'", '"')
                maybe_json_str = maybe_json_str.replace("json", "")
                # json.loads(maybe_json_str) # don't check here, cannot validate schema after all
                return json.loads(maybe_json_str)
            
        except Exception:
            print("Error: unable to locate JSON string body from string - gui_tool.py:67")
            return None
    

    def execute_each_step(self, error_message=None):

        """执行单步 - code - output"""
        if not hasattr(self, 'pipeline_steps') or not self.pipeline_steps:
            messagebox.showinfo("提示", "没有可执行的步骤")
            return
        
        #if self.current_step_index >= len(self.pipeline_steps):
            #messagebox.showinfo("提示", "所有步骤已显示完成")
            #self.single_step_btn.config(state=tk.DISABLED)
            #return


        # 当前查询
        current_query = self.running_query_display.get("1.0", tk.END).strip()
        current_prompt = self.react_solver.get_prompt(current_query,'each_step') 
        


        #????????????????????????????????????????????????????????????
        if error_message is None:
            text = self.react_solver.response(current_prompt,'complex').strip()
            print('-------text:', text)

        else :
            # 基础是当前查询
            append_query = current_query
            if hasattr(self, 'previous_output') and self.previous_output:
                append_query = f"{current_query}\n\n上一步输出:\n{self.previous_output}"

            current_prompt = self.react_solver.get_prompt(append_query,'each_step') 
            #print('??????current_prompt:', current_prompt)

            msg = "Task Information: \n" + current_prompt + \
                "When runing the script, the following error occurred: \n" + error_message + \
                "\nPlease modify the script, command.  "
            
            text = self.react_solver.response(msg,'complex').strip()

            print('??????msg:', msg)
            print('??????text_after_error:', text)

        
        #????????????????????????????????????????????????????????????
        #text = self.react_solver.response(current_prompt,'complex').strip()
        #print('??????text:', text)

        response = self.locate_json_string_body_from_string(text)
        print('#####current_response:', response)
        
        #  输出记录
        #---------------------------------
        #self.output_text.delete("1.0", tk.END)

        if response['code']:  
                
                #timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                #output = f"=== 响应 ({timestamp}) ===\n"
                #self.output_display(output)
                output = response['code'] + "\n"
                output += "=" * 50 + "\n\n"
                output += response['command'] + "\n"
                output += "=" * 50 + "\n\n"
                self.output_display(output)

                # 保存output内容，供下一步使用
                self.previous_output = output

                # 保存脚本文件
                script_file_name = response['out_script_name']
                script_file_path = os.path.join(self.selected_folder_4_output, script_file_name)
                with open(script_file_path, 'w', encoding='utf-8') as f:
                    f.write(response['code'] )
                
                # 输出命令  
                if response['command']:
                    self.code_text.delete("1.0", tk.END)
                    command = response['command'][response['command'].find('python'):]
                    self.code_text.insert("1.0", command)

        #---------------------------------
        # output code/ command
        #self.output_text.delete("1.0", tk.END)
        #self.output_text.insert(tk.END, current_prompt)

        # 更新步骤索引
        self.current_step_index += 1

        # 如果是最后一步，更新按钮状态
        if self.current_step_index >= len(self.pipeline_steps):
            self.single_step_btn.config(state=tk.DISABLED)
            self.running_query_display.insert(tk.END, "\n" + "✓ " * 10 + " 所有步骤显示完成 " + "✓" * 10)

        return
    
  
    #####################################################################################################



    def execute_query(self):
        """执行Query（使用ReAct框架）"""
        if self.current_df is None:
            messagebox.showwarning("提示", "请先加载CSV文件")
            return
        
        query = self.query_input.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("提示", "请输入Query")
            return
        
        # 开始ReAct求解过程
        self.react_solver.start_solver_loop(self.current_df, query)
        self.status_label.config(text="求解中...", foreground="blue")


    def looks_like_command(self, text):
        """判断文本是否像命令行"""
        return text.strip().startswith('python ') 
    


    def execute_action(self):
        
        code_or_command = self.code_text.get("1.0", tk.END).strip()

        # 1. 先确定模式
        mode = 'command' if self.looks_like_command(code_or_command) else 'action'
          
        if mode == 'action':
            """执行Action代码"""
            if self.current_df is None:
                messagebox.showwarning("提示", "请先加载CSV文件")
                return
            
            code = self.code_text.get("1.0", tk.END).strip()
            if not code:
                messagebox.showwarning("提示", "请输入或生成Action代码")
                return
            

            # 检查是否在ReAct求解过程中
            if self.react_solver.is_solving and self.react_solver.waiting_for_execution:
                # 在ReAct过程中执行代码
                self.react_solver.execute_current_code()
            else:
                # 独立执行代码
                try:
                    self.output_display(f"\n=== 独立执行Action代码 ===\n{code}\n")
                    
                    # 执行代码
                    observation, output = self.code_executor.python_repl(
                        code, 
                        custom_locals={'df': self.current_df.copy(), 'pd': pd}, 
                        custom_globals=globals(), 
                        memory=self.memory
                    )
                    
                    self.output_display(f"执行结果: {observation}\n")
                    if output and output != "None":
                        self.output_display(f"输出内容:\n{output}\n")
                    
                    # 如果输出是DataFrame，更新预览
                    if isinstance(output, pd.DataFrame):
                        self.output_display(f"新DataFrame形状: {output.shape}\n前5行:\n{output.head().to_string()}\n")
                    
                except Exception as e:
                    error_msg = f"执行出错: {str(e)}"
                    self.output_display(error_msg + "\n")
                    messagebox.showerror("错误", error_msg)

        elif mode =='command': # 执行Action并保存到输出文件夹

            if not hasattr(self, 'selected_folder_4_output') or not self.selected_folder_4_output:
                messagebox.showerror("错误", "请先在右侧面板选择输出文件夹！")
                return
            
            command = self.code_text.get("1.0", tk.END).strip()

            if not command:
                messagebox.showwarning("提示", "请输入要执行的命令")
                return
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output = f"=== 命令执行结果 ({timestamp}) ===\n"
            output += f"执行命令: {command}\n\n"
            self.output_display(output)

            # 执行命令
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            # # 判断是否有错误的关键词
            has_error = any(keyword in (result.stdout + result.stderr).lower() 
                for keyword in ["error:", "no such file", "not found"])
            
            if result.returncode == 0 and not has_error:
                # 执行成功
                success_output = "执行成功:\n" 
                success_output += "This Step's Result:\n" 
                success_output += result.stdout + "\n"
                success_output += "According to this Step'result, please give the rest Steps: " + "\n"
                self.output_display(success_output)

                # 执行成功 -> 更新 
                if '执行成功:' in success_output:    
                    
                    self.output_display("下个迭代正在执行中，请等待...\n")
                    # prompt 储存
                    clean_output = success_output.replace("执行成功:\n", "")
                    self.prompt_4_save += f"\n{clean_output}\n"
                    # 原来 经过 'Query Pipeline'的结果也储存
                    # memory                
                    self.output_display("-" * 60 + "\n")
                    self.output_display("prompt_4_save...\n")
                    self.output_display(self.prompt_4_save)

                    self.query_pipeline(self.prompt_4_save) 

                else :
                    self.output_display("没有执行成功\n")
            else:

                # 执行失败
                error_output = "执行失败:\n"
                error_output += f"错误信息: {result.stderr}\n"
                error_output += "=" * 50 + "\n\n"
                self.output_display(error_output)
                
                # 在主线程中显示对话框
                def show_dialog():
                    response = messagebox.askquestion(
                        "命令执行失败",
                        f"命令执行失败:\n{result.stderr}\n\n是否重新LLM生成？",
                        icon='warning'
                    )
                    
                    # 还有问题
                    if response == 'yes':
                        self.execute_each_step(error_message=result.stderr)
                        if hasattr(self, 'previous_output'):
                            self.previous_output = None
                     
                # 在主线程中显示对话框
                self.root.after(0, show_dialog)
                
            


    #########################################################################################################
    def stop_solving(self):
        """停止求解过程"""
        if self.react_solver.is_solving:
            self.react_solver.is_solving = False
            self.react_solver.waiting_for_execution = False
            self.output_display("\n=== 用户停止求解 ===\n")
            self.status_label.config(text="已停止", foreground="red")
    

    #-----------------------------------------------------------------------------------------------------
    def output_display(self, text):
        """添加输出内容"""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
    
    def clear_output(self):
        """清除输出"""
        self.output_text.delete("1.0", tk.END)
    
    def save_result(self):
        """保存结果"""
        if not self.selected_folder_4_output:
            messagebox.showwarning("提示", "请先选择工作文件夹")
            return
        
        # 获取输出内容
        output_content = self.output_text.get("1.0", tk.END)
        
        # 保存到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.selected_folder_4_output, f"result_{timestamp}.txt")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            messagebox.showinfo("成功", f"结果已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

def main():
    root = tk.Tk()
    app = CSVQueryTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()

