import os
import time
from phi.agent import Agent
from phi.model.google import Gemini
from google.generativeai import upload_file, get_file


def initialize_agent():
    """Initialize and return the video analysis agent."""
    return Agent(
        name="Video Analyst",
        model=Gemini(id="gemini-2.0-flash-exp"),
        markdown=True,
        system_prompt="""
Task: Video Behavior Analysis

You will be provided with a video depicting mouse behavior. Your goal is to describe the video and classify the behaviors observed with sufficient detail to ensure accurate behavior identification. Use the behavior list provided below to identify and rank the top 5 behaviors most confidently observed.

Behavior List
	•	A. Approach
	•	B. Nose-to-Nose Sniffing
	•	C. Stimulus Mouse Sniffing FP Mouse
	•	D. Nose-to-Flank Sniffing
	•	E. Nose-to-Anogenital Sniffing
	•	F. Withdrawal
	•	G. FP Mouse Sniffing Excretions
	•	H. Rearing
	•	I. Grooming
	•	J. FP Mouse Mounting Stimulus Mouse
	•	K. FP Mouse Snigging Excretions
	•	L. FP Mouse Tail Rattling
	•	M. Others

Instructions
	1.	Analyze the Video:
Examine the video to identify the most dominant and relevant behaviors.
	2.	Rank Behaviors:
Rank the top 5 observed behaviors by confidence (most likely) and relevance (most descriptive). Use the behavior list above.
	3.	Justify Observations (Optional):
If clarity is required, provide a 1–2 sentence explanation for each selected behavior to support your ranking.

Output Format

1. <Behavior Code> (<Behavior Description>): <Optional Justification>  
2. <Behavior Code> (<Behavior Description>): <Optional Justification>  
...  

Example Output

1. B (Nose-to-Nose Sniffing): Direct nose-to-nose interaction is visible.  
2. A (Approach): The FP mouse approaches the stimulus mouse.  
3. D (Nose-to-Flank Sniffing): Flank sniffing observed.  
4. F (Withdrawal): The FP mouse retreats after interaction.  
5. M (Others): General movement across the enclosure.  

...

Important Notes for Classification
	•	Use context clues (e.g., proximity, positioning, and movement) to distinguish between similar behaviors.
    •	Pay close attention to mice body parts like nose, tailbase, paws and other moving body parts.
	•	If multiple behaviors occur simultaneously, prioritize based on relevance to overall interaction.
    """,
    )


def process_video_with_agent(video_path, user_prompt, agent):
    """
    Analyze a video and perform research based on the user's prompt.

    Args:
        video_path (str): Path to the video file.
        user_prompt (str): The question or prompt for the agent.
        agent (Agent): The initialized agent for video analysis and research.

    Returns:
        str: The response from the agent.
    """
    try:
        # Upload the video for processing
        print("Uploading video...")
        video_file = upload_file(video_path)
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = get_file(video_file.name)

        # Build the prompt for the agent
        prompt = f"""
        First analyze this video and then answer the following question using
        the video analysis: {user_prompt}
        
        Provide a comprehensive response focusing on practical, actionable information.
        """

        # Run the agent with the video and prompt
        print("Analyzing video and performing research...")
        result = agent.run(prompt, videos=[video_file])
        return result.content

    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    # Example usage
    # Replace with your video file path
    video_path = os.path.expanduser(
        "~/Downloads/Stimulus Mouse Sniffing FP Mouse_3.16-7.64.mp4")
    user_prompt = """Describe the main activities and classify behavior in this video."""

    # Initialize the agent
    agent = initialize_agent()

    # Process the video and get the result
    response = process_video_with_agent(video_path, user_prompt, agent)
    print("Agent Response:")
    print(response)
