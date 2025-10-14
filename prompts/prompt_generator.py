from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    template = """
        You are an expert researcher and educator. Your task is to explain the research paper titled "{selected_paper}" to a user. 
        The explanation should follow these requirements:

        1. **Explanation Style:** {selected_style}  
       

        2. **Explanation Length:** {selected_length}  
        - Short: Provide a concise summary highlighting the main ideas.  
        - Medium: Include explanations of key methods, results, and insights with examples.  
        - Long: Provide a comprehensive walkthrough, covering background, methodology, results, implications, and optionally code or math depending on the style.

        3. **Structure (Optional, if applicable):**
        - Introduction: Context and motivation behind the paper.  
        - Problem Statement: Clearly define the problem being solved.  
        - Methodology: Describe the techniques, models, or algorithms used.  
        - Results: Summarize key findings or experiments.  
        - Conclusion: Implications, applications, and limitations.

        4. Use clear, coherent language. Adapt examples or explanations according to the selected style and depth. Make it engaging and easy to follow while being accurate.

        5. If you dont have information, simply say i dont know insteead of guessing

        Now, generate the explanation.

""",
   input_variables = ['selected_paper', 'selected_style', 'selected_length']
)
  
template.save('template.json')  