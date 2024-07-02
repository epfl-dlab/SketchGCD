import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from src.prompter import BasePrompter, DualInputPrompter
from src.const import PROMPTER_DIR


if __name__ == "__main__":
    # Demo needs to be a list of dictionaries with 'input' and 'output' keys
    demo_pool = [
        {
            "input": "Kintla Glacier is in Glacier National Park in the U.S. state of Montana.",
            "draft": "[s] Kintla Glacier [r] located in [o] Glacier National Park [e] [s] Kintla Glacier [r] located in [o] U.S. state of Montana [e]",
            "output": "[s] Kintla Glacier [r] located in the administrative territorial entity [o] Montana [e]",
        },
        {
            "input": "Lecques is a commune in the Gard department in southern France .",
            "draft": "[s] Lecques [r] located in the administrative territorial entity [o] Gard department [e] [s] Lecques [r] located in the administrative territorial entity [o] France [e]",
            "output": "[s] Lecques [r] located in the administrative territorial entity [o] Gard [e] [s] Lecques [r] country [o] France [e]",
        },
        {
            "input": "Gwen John was born in Haverfordwest , Wales , the second of four children of Edwin William John and Gwen John wife Augusta .",
            "draft": "[s] Gwen John [r] place of birth [o] Haverfordwest [e] [s] Gwen John [r] child [o] Edwin William John [e] [s] Gwen John [r] child [o] Gwen John wife Augusta [e]",
            "output": "[s] Gwen John [r] place of birth [o] Haverfordwest [e] [s] Gwen John [r] country of citizenship [o] Wales [e]",
        },
        {
            "input": "Allen Dale June , along with the other original 9 Navajo code talkers , received the Congressional Gold Medal on December 21 , 2000 .",
            "draft": "[s] Allen Dale June [r] recipient of [o] Congressional Gold Medal [e]",
            "output": "[s] Allen Dale June [r] award received [o] Congressional Gold Medal [e]",
        },
        {
            "input": "Grosberg died on July 25 , 1970 at Mount Sinai Hospital in Miami Beach, Florida ; Joseph E. Grosberg was 86 .",
            "draft": "[s] Grosberg [r] date of death [o] July 25, 1970 [e] [s] Grosberg [r] place of death [o] Mount Sinai Hospital [e] [s] Grosberg [r] location [o] Miami Beach, Florida [e] [s] Joseph E. Grosberg [r] age [o] 86 [e]",
            "output": "[s] Joseph E. Grosberg [r] place of death [o] Miami Beach, Florida [e]",
        },
        {
            "input": "Alexander Gauland was born in 1941 in Chemnitz , a city that became part of East Germany in 1949 and was renamed Karl - Marx - Stadt .",
            "draft": "[s] Alexander Gauland [r] place of birth [o] Chemnitz [e] [s] Chemnitz [r] part of [o] East Germany [e] [s] Chemnitz [r] renamed [o] Karl-Marx-Stadt [e]",
            "output": "[s] Alexander Gauland [r] place of birth [o] Chemnitz [e] [s] Alexander Gauland [r] country of citizenship [o] East Germany [e]",
        },
    ]
    # runtime just has 'input' key
    runtime_input = {"input": "What is the capital of France?", "draft": "Paris"}

    # Check if random selection is working
    wikinre_fe_prompter = DualInputPrompter(
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

    file_path = os.path.join(PROMPTER_DIR, "IE", "wikinre_fe_Q_few.json")

    wikinre_fe_prompter.save(file_path)

    BasePrompter.from_local(file_path).pretty_print()

    # genie

    demo_pool = [
        {
            "input": "The Artemis Accords are a set of agreements created by NASA and valid in outer space, with Australia, Ukraine, and Colombia as signatories. Outer space is used for spaceflight, which is the opposite of astronomical objects.The Artemis Accords are a set of agreements created by NASA and valid in outer space, with Australia, Ukraine, and Colombia as signatories. Outer space is used for spaceflight, which is the opposite of astronomical objects.",
            "draft": "[s] Artemis_Accords [r] created by [o] NASA [e] [s] Artemis_Accords [r] applicable jurisdiction [o] Outer_space [e] [s] Artemis_Accords [r] signatory [o] Australia [e] [s] Artemis_Accords [r] signatory [o] Ukraine [e] [s] Artemis_Accords [r] signatory [o] Colombia [e] [s] Outer_space [r] used for [o] Spaceflight [e] [s] Spaceflight [r] opposite of [o] Astronomical_objects [e]",
            "output": "[s] Artemis_Accords [r] creator [o] NASA [e] [s] Artemis_Accords [r] valid in place [o] Outer_space [e] [s] Artemis_Accords [r] signatory [o] Australia [e] [s] Artemis_Accords [r] signatory [o] Ukraine [e] [s] Artemis_Accords [r] signatory [o] Colombia [e] [s] outer_space [r] use [o] Spaceflight [e] [s] Outer_space [r] opposite of [o] Astronomical_object [e]",
        },
        {
            "input": "Vettaikaaran (2009 film) was originally written in the Tamil language, with B. Babusivan as the screenwriter.",
            "draft": "[s] Vettaikaaran_(2009_film) [r] language of work or name [o] Tamil_language [e] [s] Vettaikaaran_(2009_film) [r] screenwriter [o] B._Babusivan [e]",
            "output": "[s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [e] [s] Vettaikaaran_(2009_film) [r] screenwriter [o] B._Babusivan [e]",
        },
        {
            "input": "Abhishek Pictures is a film production company based in Hyderabad.",
            "draft": "[s] Abhishek_Pictures [r] subclass of [o] Film_production_company [e] [s] Abhishek_Pictures [r] located in the administrative territorial entity [o] Hyderabad [e]",
            "output": "[s] Abhishek_Pictures [r] industry [o] Film_industry [e] [s] Abhishek_Pictures [r] headquarters location [o] Hyderabad [e]",
        },
        {
            "input": "Swedish Open Cultural Heritage is a project developed by the Swedish National Heritage Board, which is mainly focused on cultural heritage. It produces Resource Description Framework as its product or material and uses XML, JSON, and JSON-LD as its file formats. XML was inspired by Standard Generalized Markup Language.",
            "draft": "[s] Swedish_Open_Cultural_Heritage [r] developed by [o] Swedish_National_Heritage_Board [e] [s] Swedish_Open_Cultural_Heritage [r] focus [o] Cultural_heritage [e] [s] Swedish_Open_Cultural_Heritage [r] product or material produced [o] Resource_Description_Framework [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] XML [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON-LD [e] [s] XML [r] inspired by [o] Standard_Generalized_Markup_Language [e]",
            "output": "[s] Swedish_Open_Cultural_Heritage [r] developer [o] Swedish_National_Heritage_Board [e] [s] Swedish_Open_Cultural_Heritage [r] main subject [o] Cultural_heritage [e] [s] Swedish_Open_Cultural_Heritage [r] product or material produced [o] Resource_Description_Framework [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] XML [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON-LD [e] [s] XML [r] inspired by [o] Standard_Generalized_Markup_Language [e]",
        },
        {
            "input": "Marcus Jacob Monrad was a Lutheran who passed away in Oslo and was buried in the Cemetery of Our Saviour.Marcus Jacob Monrad was a Lutheran who passed away in Oslo and was buried in the Cemetery of Our Saviour.",
            "draft": "[s] Marcus_Jacob_Monrad [r] religion [o] Lutheran [e] [s] Marcus_Jacob_Monrad [r] place of death [o] Oslo [e] [s] Marcus_Jacob_Monrad [r] place of burial [o] Cemetery_of_Our_Saviour [e]",
            "output": "[s] Marcus_Jacob_Monrad [r] religion [o] Lutheranism [e] [s] Marcus_Jacob_Monrad [r] place of death [o] object [e] [s] Marcus_Jacob_Monrad [r] place of burial [o] Cemetery_of_Our_Saviour [e]",
        },
        {
            "input": "Papaver rhoeas is a species of plant that is invasive to China and has a variety of uses, such as the production of fruit.",
            "draft": "[s] Papaver_rhoeas [r] taxon rank [o] Species [e] [s] Papaver_rhoeas [r] invasive to [o] China [e] [s] Papaver_rhoeas [r] has use [o] Fruit_production [e]",
            "output": "[s] Papaver_rhoeas [r] taxon rank [o] Species [e] [s] Papaver_rhoeas [r] invasive to [o] China [e] [s] Papaver_rhoeas [r] use [o] Fruit [e] [s] Papaver_rhoeas [r] instance of [o] Taxon [e]",
        },
        {
            "input": "Gymnastics at the 2006 Commonwealth Games was a sport held in Australia as part of the 2006 Commonwealth Games, an instance of Season (sports). Australia has a diplomatic relation with Mauritius.",
            "draft": "[s] Gymnastics_at_the_2006_Commonwealth_Games [r] country [o] Australia [e] [s] Gymnastics_at_the_2006_Commonwealth_Games [r] instance of [o] Season_sports [e] [s] Australia [r] diplomatic relations [o] Mauritius [e] [s] Gymnastics_at_the_2006_Commonwealth_Games [r] part of [o] 2006_Commonwealth_Games [e]",
            "output": "[s] Gymnastics_at_the_2006_Commonwealth_Games [r] sport [o] Gymnastics [e] [s] Gymnastics_at_the_2006_Commonwealth_Games [r] part of [o] 2006_Commonwealth_Games [e] [s] Gymnastics_at_the_2006_Commonwealth_Games [r] country [o] Australia [e] [s] Gymnastics_at_the_2006_Commonwealth_Games [r] instance of [o] Season_(sports) [e] [s] Australia [r] diplomatic relation [o] Mauritius [e]",
        },
    ]

    # Check if random selection is working
    synthie_fe_prompter = DualInputPrompter(
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

    file_path = os.path.join(PROMPTER_DIR, "IE", "synthie_fe_Q.json")

    synthie_fe_prompter.save(filename=file_path)

    BasePrompter.from_local(file_path).pretty_print()
