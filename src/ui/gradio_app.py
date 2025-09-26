import gradio as gr 
from src.ui.tabs import config_tab, cabinets_tab, evaluate_tab, output_tab

def build_demo():
    with gr.Blocks(title="Store DR Unloadability Console") as demo:
        gr.Markdown("# 門市卸載評估控制台")
        with gr.Tabs():
            config_tab.render()
            evaluate_tab.render()   # ← 新增的「設備卸載評估」
            df = cabinets_tab.render()
            output_tab.render(df)   # ← 原本 evaluate_tab 改名為 output_tab
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
