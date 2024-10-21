flan_fewshot_tqa = """Which British politician was the first person to be made an Honorary Citizen of the United States of America?  
Answer: Winston Churchill  
  
Which event of 1962 is the subject of the 2000 film Thirteen Days starring Kevin Costner?  
Answer: The Cuban Missile Crisis  
  
Which country hosted the 1968 Summer Olympics?  
Answer: Mexico  
  
In which city did the assassination of Martin Luther King?  
Answer: MEMPHIS, Tennessee  
  
Which German rye bread is named, according to many reliable sources, from the original meaning 'Devil's fart'?  
Answer: Pumpernickel"""

flan_fewshot_nq = """who won a million on deal or no deal
Answer: Tomorrow Rodriguez

who is the woman washing the car in cool hand luke
Answer: Joy Harmon

who is the actor that plays ragnar on vikings
Answer: Travis Fimmel

who said it's better to have loved and lost
Answer: Alfred , Lord Tennyson

name the first indian woman to be crowned as miss world
Answer: Reita Faria"""

flan_fewshot_hotpot = """Which magazine was started first Arthur's Magazine or First for Women?
Answer: Arthur's Magazine

The Oberoi family is part of a hotel company that has a head office in what city?
Answer: Delhi

Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Answer: President Richard Nixon

What nationality was James Henry Miller's wife?
Answer: American

Cadmium Chloride is slightly soluble in this chemical, it is also called what?
Answer: alcohol
"""
flan_fewshot_musique = """Who was ordered to force a Tibetan assault into the region conquered by Yellow Tiger in the mid-17th century?
Answer: Ming general Qu Neng

What date was the start of the season of Grey's Anatomy where Eric died?
Answer: September 25, 2014

When did the publisher of Tetrisphere unveil their new systems?
Answer: October 18, 1985

Who is the composer of Rhapsody No. 1, named after and inspired by the county where Alfred Seaman was born?
Answer: Ralph Vaughan Williams

What region is Qaleh Now-e Khaleseh in Mahdi Tajik's birth city located?
Answer: Qaleh Now Rural District"""

llama_fewshot_tqa = """Question: Which British politician was the first person to be made an Honorary Citizen of the United States of America?  
Answer: Winston Churchill  
  
Question: Which event of 1962 is the subject of the 2000 film Thirteen Days starring Kevin Costner?  
Answer: The Cuban Missile Crisis  
  
Question: Which country hosted the 1968 Summer Olympics?  
Answer: Mexico  
  
Question: In which city did the assassination of Martin Luther King?  
Answer: MEMPHIS, Tennessee  
  
Question: Which German rye bread is named, according to many reliable sources, from the original meaning 'Devil's fart'?  
Answer: Pumpernickel"""

llama_fewshot_nq = """Question: who won a million on deal or no deal
Answer: Tomorrow Rodriguez

Question: who is the woman washing the car in cool hand luke
Answer: Joy Harmon

Question: who is the actor that plays ragnar on vikings
Answer: Travis Fimmel

Question: who said it's better to have loved and lost
Answer: Alfred , Lord Tennyson

Question: name the first indian woman to be crowned as miss world
Answer: Reita Faria"""

llama_fewshot_hotpot = """Question: Which magazine was started first Arthur's Magazine or First for Women?
Answer: Arthur's Magazine

Question: The Oberoi family is part of a hotel company that has a head office in what city?
Answer: Delhi

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Answer: President Richard Nixon

Question: What nationality was James Henry Miller's wife?
Answer: American

Question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
Answer: alcohol"""


llama_fewshot_musique = """Question: Who was ordered to force a Tibetan assault into the region conquered by Yellow Tiger in the mid-17th century?
Answer: Ming general Qu Neng

Question: What date was the start of the season of Grey's Anatomy where Eric died?
Answer: September 25, 2014

Question: When did the publisher of Tetrisphere unveil their new systems?
Answer: October 18, 1985

Question: Who is the composer of Rhapsody No. 1, named after and inspired by the county where Alfred Seaman was born?
Answer: Ralph Vaughan Williams

Question: What region is Qaleh Now-e Khaleseh in Mahdi Tajik's birth city located?
Answer: Qaleh Now Rural District"""


general_prompt = """
{few_shot}

{document}
{question}
Answer:
"""

general_llama_prompt="""
{few_shot}

{document}
Question: {question}
Answer:
"""


# System prompt
# See discussion here:
# https://github.com/EleutherAI/lm-evaluation-harness/issues/1292
job_description = "Answer these questions:\n\n"