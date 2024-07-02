import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from src.prompter import BasePrompter
from src.const import PROMPTER_DIR


if __name__ == "__main__":
    # Demo needs to be a list of dictionaries with 'input' and 'output' keys
    demo_pool = [
        {
            "input": "[s] Kintla Glacier [r] located in the administrative territorial entity [o] Glacier National Park [e] [s] Kintla Glacier [r] located in the administrative territorial entity [o] Montana [e]",
            "output": "[s] Kintla Glacier [r] located in the administrative territorial entity [o] Montana [e]",
        },
        {
            "input": "[s] Lecques [r] located in the administrative territorial entity [o] Gard department [e] [s] Lecques [r] located in the administrative territorial entity [o] France [e]",
            "output": "[s] Lecques [r] located in the administrative territorial entity [o] Gard department [e] [s] Lecques [r] country [o] France [e]",
        },
        {
            "input": "[s] Gwen John [r] place of birth [o] Haverfordwest [e] [s] Gwen John [r] country of citizenship [o] Wales [e] [s] Gwen John [r] father [o] Edwin William John [e] [s] Gwen John [r] mother [o] Augusta [e]",
            "output": "[s] Gwen John [r] place of birth [o] Haverfordwest [e] [s] Gwen John [r] country of citizenship [o] Wales [e]",
        },
        {
            "input": "[s] Besham [r] located in the administrative territorial entity [o] Shangla District [e] [s] Besham [r] located in the administrative territorial entity [o] Khyber - Pakhtunkhwa [e] [s] Besham [r] country [o] Pakistan [e]",
            "output": "[s] Besham [r] country [o] Pakistan [e]",
        },
        {
            "input": "[s] Alex Lora [r] place of birth [o] Barcelona [e] [s] Alex Lora [r] country of birth [o] Spain [e]",
            "output": "[s] Alex Lora Cercos [r] place of birth [o] Barcelona [e] [s] Alex Lora Cercos [r] country of citizenship [o] Spain [e]",
        },
        {
            "input": "[s] Ramakrishna Badiga [r] date of birth [o] 2 September 1942 [e] [s] Ramakrishna Badiga [r] member of [o] 14th Lok Sabha [e]",
            "output": "[s] Ramakrishna Badiga [r] country of citizenship [o] India [e]",
        },
    ]
    # runtime just has 'input' key
    runtime_input = {"input": "What is the capital of France?"}

    # Check if random selection is working
    wikinre_fe_prompter = BasePrompter(
        context="",
        task="In this task, you will be provided with a draft annotations that represent information extraction in the form of triples (subject, relation, object) from a given text. Your task is to correct the annotations to make them correct.",
        instruction="NO MORE THAN TWO TRIPLE PER TEXT.",
        demo_pool=demo_pool,
        num_demo=4,
        seed=42,
        demo_selection="first",
        output_prefix=' """',
    )

    wikinre_fe_prompter.pretty_print()

    prompt = wikinre_fe_prompter.materialize(runtime_input)
    print(prompt)

    file_path = os.path.join(PROMPTER_DIR, "IE", "wikinre_fe_A_few.json")

    wikinre_fe_prompter.save(file_path)

    BasePrompter.from_local(file_path).pretty_print()

    # genie

    demo_pool = [
        {
            "input": "[s] Artemis_Accords [r] created by [o] NASA [e] [s] Artemis_Accords [r] applicable jurisdiction [o] Outer_space [e] [s] Artemis_Accords [r] signatory [o] Australia [e] [s] Artemis_Accords [r] signatory [o] Ukraine [e] [s] Artemis_Accords [r] signatory [o] Colombia [e] [s] Outer_space [r] used for [o] Spaceflight [e] [s] Spaceflight [r] opposite of [o] Astronomical_objects [e]",
            "output": "[s] Artemis_Accords [r] creator [o] NASA [e] [s] Artemis_Accords [r] valid in place [o] Outer_space [e] [s] Artemis_Accords [r] signatory [o] Australia [e] [s] Artemis_Accords [r] signatory [o] Ukraine [e] [s] Artemis_Accords [r] signatory [o] Colombia [e] [s] outer_space [r] use [o] Spaceflight [e] [s] Outer_space [r] opposite of [o] Astronomical_object [e]",
        },
        {
            "input": "[s] Vettaikaaran_(2009_film) [r] language of work or name [o] Tamil_language [e] [s] Vettaikaaran_(2009_film) [r] screenwriter [o] B._Babusivan [e]",
            "output": "[s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [e] [s] Vettaikaaran_(2009_film) [r] screenwriter [o] B._Babusivan [e]",
        },
        {
            "input": "[s] Abhishek_Pictures [r] subclass of [o] Film_production_company [e] [s] Abhishek_Pictures [r] located in the administrative territorial entity [o] Hyderabad [e]",
            "output": "[s] Abhishek_Pictures [r] industry [o] Film_industry [e] [s] Abhishek_Pictures [r] headquarters location [o] Hyderabad [e]",
        },
        {
            "input": "[s] Swedish_Open_Cultural_Heritage [r] developed by [o] Swedish_National_Heritage_Board [e] [s] Swedish_Open_Cultural_Heritage [r] focus [o] Cultural_heritage [e] [s] Swedish_Open_Cultural_Heritage [r] product or material produced [o] Resource_Description_Framework [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] XML [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON-LD [e] [s] XML [r] inspired by [o] Standard_Generalized_Markup_Language [e]",
            "output": "[s] Swedish_Open_Cultural_Heritage [r] developer [o] Swedish_National_Heritage_Board [e] [s] Swedish_Open_Cultural_Heritage [r] main subject [o] Cultural_heritage [e] [s] Swedish_Open_Cultural_Heritage [r] product or material produced [o] Resource_Description_Framework [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] XML [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON-LD [e] [s] XML [r] inspired by [o] Standard_Generalized_Markup_Language [e]",
        },
        {
            "input": "[s] Marcus_Jacob_Monrad [r] religion [o] Lutheran [e] [s] Marcus_Jacob_Monrad [r] place of death [o] Oslo [e] [s] Marcus_Jacob_Monrad [r] place of burial [o] Cemetery_of_Our_Saviour [e]",
            "output": "[s] Marcus_Jacob_Monrad [r] religion [o] Lutheranism [e] [s] Marcus_Jacob_Monrad [r] place of death [o] object [e] [s] Marcus_Jacob_Monrad [r] place of burial [o] Cemetery_of_Our_Saviour [e]",
        },
    ]

    # runtime just has 'input' key
    runtime_input = {"input": "What is the capital of France?"}

    # Check if random selection is working
    synthie_fe_prompter = BasePrompter(
        context="",
        task="In this task, you will be provided with a draft annotations that represent information extraction in the form of triples (subject, relation, object) from a given text. Your task is to correct the annotations to make them correct.",
        instruction="Do not add or remove any triples. You can only change the triples that are already there.",
        demo_pool=demo_pool,
        num_demo=4,
        seed=42,
        demo_selection="first",
        output_prefix=' """',
    )

    synthie_fe_prompter.pretty_print()

    prompt = synthie_fe_prompter.materialize(runtime_input)
    print(prompt)

    file_path = os.path.join(PROMPTER_DIR, "IE", "synthie_fe_A.json")

    synthie_fe_prompter.save(filename=file_path)

    BasePrompter.from_local(file_path).pretty_print()
