"""Templates for generating prompts for genetic testing recommendations."""

from __future__ import annotations

from liquid import Template

general_cot_system = (
    'You are a clinical expert on rare genetic diseases. Does this '
    'patient need genetic testing for a potential undiagnosed rare genetic '
    'disease or syndrome based on their past and present symptoms and medical '
    'history? If yes, provide specific criteria why, otherwise state why the '
    'patient does not require genetic testing. Return your response as a JSON '
    "formatted string with two parts 1) testing recommendation ('recommended' "
    "or 'not recommended') and 2) your reasoning in the format as follows:\n\n"
    '{\n'
    '  "testing_recommendation": "[recommended or not recommended]",\n'
    '  "reasoning": "[Your detailed reasoning based on clinical summary]"\n'
    '}'
)

general_cot = Template("""
Here is the question:
{{question}}

Please think step-by-step and generate your output in json:
""")

general_medrag_system = (
    'You are a clinical expert on rare genetic diseases. Does this patient '
    'need genetic testing for a potential undiagnosed rare genetic disease '
    'or syndrome based on their past and present symptoms and medical '
    'history? If yes, provide specific criteria why, otherwise state why '
    'the patient does not require genetic testing. Return your response as '
    'a JSON formatted string with two parts 1) testing recommendation '
    "('recommended' or 'not recommended') and 2) your reasoning in the "
    'format as follows:\n\n'
    '{\n'
    '  "testing_recommendation": "[recommended or not recommended]",\n'
    '  "reasoning": "[Your detailed reasoning based on clinical summary]"\n'
    '}'
)

general_medrag = Template("""
Here is the patient's clinical note:
{{question}}

Here are relevant medical literatures you can use to assess whether the
patient needs genetic testing:
{{context}}

Please think step-by-step and generate your output in json:
""")

system_prompt_1 = (
    'You are a clinical expert on rare genetic diseases. Does this patient '
    'need genetic testing for a potential undiagnosed rare genetic disease '
    'or syndrome based on their past and present symptoms and medical '
    'history? If yes, provide specific criteria why, otherwise state why '
    'the patient does not require genetic testing. Return your response as '
    'a JSON formatted string with two parts 1) testing recommendation '
    "('recommended' or 'not recommended') and 2) your reasoning in the "
    'format as follows:\n\n'
    '{\n'
    '  "testing_recommendation": "[recommended or not recommended]",\n'
    '  "reasoning": "[Your detailed reasoning based on clinical summary]"\n'
    '}'
)

system_prompt_2 = (
    'You are a clinical expert on rare genetic diseases. Does this patient '
    'need genetic testing for a potential undiagnosed rare genetic disease '
    'or syndrome based on their past and present symptoms and medical '
    "history? Return your response as a JSON formatted string. 'Testing "
    "Recommendation': ['Recommended' or 'Not Recommended']"
)
