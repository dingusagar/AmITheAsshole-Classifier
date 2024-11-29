from transformers import pipeline, DistilBertTokenizerFast

# Path to your locally saved model
# bert_model_path = "fine_tuned_aita_classifier"
bert_model_path = "dingusagar/distillbert-aita-classifier"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
# tokenizer.save_pretrained(bert_model_path)

classifier = pipeline(
    "text-classification",
    model=bert_model_path,  # Path to your locally saved model
    tokenizer=tokenizer,  # Use the tokenizer saved with the model
    truncation=True
)

bert_label_map = {
    'LABEL_0': 'YTA',
    'LABEL_1': 'NTA',
}


def ask_bert(prompt):
    result = classifier([prompt])[0]
    label = bert_label_map.get(result['label'])
    confidence = f"{result['score']:.4f}"
    return label, confidence




# Example input text
texts = [
    "AITA for not forgetting to invite my best friend for my marriage ?",
    "AITA for accidentally getting the name wrong for a person I met twice ?",
    "AITA saving someone for no reason ?",
    """AITA for telling my friend I won't be inviting her out anymore? So my bf tells me im not the asshole but I feel like I may be. So I F23 have a friend I'll call Mary who's 22. She and I work together and became friends. Well this past Saturday, I invited her out with my friend group to go to a local amusement park that goes all out for Halloween

Mary asks if she can bring her kids a 5 year old boy and an 18 month old girl. Everyone involved tells her it's really not a good idea as this park and it's haunted attractions are not geared towards children and we're planning on being there until it closes which is midnight.

She seems to accept this but asks repeatedly throughout the week leading up to Saturday and she is again told no. Well Saturday arrived and as you can guess, she brought her kids. Other people in our group asks her why and she just shrugged saying she thought the kids would have fun.

They didn't, her son got scared with in about 10 minutes of us getting into the park and began to cry begging to go home to which Mary tells him to calm down and he'll have fun eventually. We get in line for the first haunted house and her son again starts to cry saying he doesn't want to go into the house. Mary then asks myself if I'll stay and watch her son and daughter so she can go into the haunted house.

I tell her no and that this is why we told her not to bring her kids. She gets upset and drags her very scared child through the haunted house. He had a melt down and had to be carried out. This repeats through every single haunted house we attempted to go through.

Around 11:30, my boyfriend pulls me aside and tells me that he can't take anymore of the screaming/crying and we try to break off to find a place to calm down, Mary sees this and leaves her son and daughter with us while she runs off to go on a ride. Her son gets scared by an actor chasing people with a chainsaw and has an epic melt down. I'm doing the best I can to console him but I am rapidly running out of patients. Finally his mom comes back and I all but shove her son back into her arms

I tell Mary that my bf and I were leaving along with the rest of our group. She gets huffy but agreed. We leave the park and go to waffle House for dinner. At this point it's midnight and both kids are extremely tired and upset. They cry all through dinner and Mary did nothing to calm them down. Finally at the end of my rope, once we get out of the restaurant I lose my temper. I tell Mary that this is why she was told not to bring her kids to this event and that I will not be inviting her back out again if she can't follow the rules of the group. Mary got upset and has since blocked me and the other people who agreeded with me. No one in the group agreed with Mary but they all did say that I didn't need to say anything about it to her and I didn't need to tell her I wasn't inviting her out again."""

]

res = ask_bert(texts[-1])
print(res)
