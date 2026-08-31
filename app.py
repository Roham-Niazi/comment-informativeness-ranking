import gradio as gr
from src.ranker import CommentRanker
import os

#Function to prepare sample text
def set_samples_btn_click(lang):
	if lang=="English":
		filename="examples/sample_en.txt"
	elif lang=="Persian":
		filename="examples/sample_fa.txt"

	with open(filename, "r", encoding="utf-8") as f:
		comments=f.read()

	return comments


def submit_comments(comments, lang):
	languages_dict={
		"English": "en",
		"Persian": "fa"
	}

	#Getting the ranking results
	output=CommentRanker(languages_dict[lang]).rank(comments.splitlines())

	#Unpacking the ranking results
	comments_lst, scores=list(zip(*output))

	#Preparing data to show in table
	data=list(zip(
		range(1, len(comments_lst)+1),
		comments_lst,
		map(lambda x: f"{x:.5f}", scores)
	))
	return data



with gr.Blocks() as demo:
	with gr.Row():
		with gr.Column():
			comments_input=gr.Textbox(label="Comments", placeholder="Enter each comment in one line", lines=10)

			dropdown=gr.Dropdown(["English", "Persian"], value="English", label="Language")

			with gr.Row():
				set_samples_btn=gr.Button("Try Sample")
				submit_btn=gr.Button("Submit", variant="primary")

		with gr.Column():
			table=gr.Dataframe(
				label="Ranked Comments",
				headers=["Rank", "Comment", "Score"],
				elem_id="output_table"
			)
	

	set_samples_btn.click(
		fn=set_samples_btn_click,
		inputs=dropdown,
		outputs=comments_input,
	)
	submit_btn.click(
		fn=submit_comments,
		inputs=[comments_input, dropdown],
		outputs=table,
	)



demo.launch(
	server_name="0.0.0.0",
	server_port=int(os.environ.get("PORT", 7860)),
	share=False
)