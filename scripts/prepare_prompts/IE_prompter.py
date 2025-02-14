import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from src.prompter import BasePrompter
from src.const import PROMPTER_DIR


if __name__ == "__main__":
    # Demo needs to be a list of dictionaries with 'input' and 'output' keys
    demo_pool = [
        {
            "input": "Born in Cologne , Daniel Baier first played for the youth team of 1860 Munich .",
            "output": "[s] Daniel_Baier [r] place of birth [o] Cologne [e] [end]",
        },
        {
            "input": "Masami Tachikawa is a Japanese basketball player who competed in the 2004 Summer Olympics .",
            "output": "[s] Masami_Tachikawa [r] country of citizenship [o] Japan [e] [s] Masami_Tachikawa [r] participant in [o] 2004 Summer_Olympics [e] [end]",
        },
        {
            "input": "County Leitrim was long influenced by the O'Rourke family of Dromahair , whose heraldic lion occupies the official county shield to this day .",
            "output": "[s] Dromahair [r] located in the administrative territorial entity [o] County_Leitrim [e] [end]",
        },
        {
            "input": "Wolves of the Rail is a 1918 American silent western film produced , directed by , and starring William S. Hart .",
            "output": "[s] Wolves_of_the_Rail [r] cast member [o] William_S._Hart [e] [s] Wolves_of_the_Rail [r] director [o] William_S._Hart [e] [end]",
        },
    ]
    # runtime just has 'input' key
    runtime_input = {"input": "What is the capital of France?"}

    # Check if random selection is working
    wikinre_fe_prompter = BasePrompter(
        context="",
        task="",
        instruction="Extract the triples in fully-expanded format from texts below.",
        demo_pool=demo_pool,
        num_demo=4,
        seed=42,
        demo_selection="first",
        output_prefix=' """',
    )

    wikinre_fe_prompter.pretty_print()

    prompt = wikinre_fe_prompter.materialize(runtime_input)
    print(prompt)

    file_path = os.path.join(PROMPTER_DIR, "IE", "wikinre_fe.json")

    wikinre_fe_prompter.save(file_path)

    BasePrompter.from_local(file_path).pretty_print()

    # genie

    demo_pool = [
        {
            "input": "Vettaikaaran (2009 film) was originally written in the Tamil language, with B. Babusivan as the screenwriter.",
            "output": "[s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [e] [s] Vettaikaaran_(2009_film) [r] screenwriter [o] B._Babusivan [e] [end]",
        },
        {
            "input": "The NHL Stadium Series is a sport that consists of ice hockey.",
            "output": "[s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e] [end]",
        },
        {
            "input": "Abhishek Pictures is a film production company based in Hyderabad.",
            "output": "[s] Abhishek_Pictures [r] industry [o] Film_industry [e] [s] Abhishek_Pictures [r] headquarters location [o] Hyderabad [e] [end]",
        },
        {
            "input": "Swedish Open Cultural Heritage is a project developed by the Swedish National Heritage Board, which is mainly focused on cultural heritage. It produces Resource Description Framework as its product or material and uses XML, JSON, and JSON-LD as its file formats. XML was inspired by Standard Generalized Markup Language.",
            "output": "[s] Swedish_Open_Cultural_Heritage [r] main subject [o] Cultural_heritage [e] [s] Swedish_Open_Cultural_Heritage [r] developer [o] Swedish_National_Heritage_Board [e] [s] Swedish_Open_Cultural_Heritage [r] product or material produced [o] Resource_Description_Framework [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] XML [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON [e] [s] Swedish_Open_Cultural_Heritage [r] file format [o] JSON-LD [e] [s] XML [r] inspired by [o] Standard_Generalized_Markup_Language [e] [end]",
        },
        {
            "input": "The General Administration of Quality Supervision, Inspection and Quarantine was replaced by the State Administration for Market Regulation and is a government agency under the parent organization, the State Council of the People's Republic of China. Its headquarters is located in Haidian District, China.",
            "output": "[s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] replaced by [o] State_Administration_for_Market_Regulation [e] [s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] instance of [o] Government_agency [e] [s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] parent organization [o] State_Council_of_the_People's_Republic_of_China [e] [s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] country [o] China [e] [s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] headquarters location [o] Haidian_District [e] [end]",
        },
        {
            "input": "Marcus Jacob Monrad was a Lutheran who passed away in Oslo and was buried in the Cemetery of Our Saviour.Marcus Jacob Monrad was a Lutheran who passed away in Oslo and was buried in the Cemetery of Our Saviour.",
            "output": "[s] Marcus_Jacob_Monrad [r] place of death [o] object [e] [s] Marcus_Jacob_Monrad [r] place of burial [o] Cemetery_of_Our_Saviour [e] [s] Marcus_Jacob_Monrad [r] religion [o] Lutheranism [e] [end]",
        },
        {
            "input": "The Making of Maddalena was shot by James Van Trees as director of photography and is presented in black and white. The cast includes Edna Goodrich.",
            "output": "[s] The_Making_of_Maddalena [r] director of photography [o] James_Van_Trees [e] [s] The_Making_of_Maddalena [r] color [o] black_and_white [e] [s] The_Making_of_Maddalena [r] cast member [o] Edna_Goodrich [e] [end]",
        },
        {
            "input": "The Artemis Accords are a set of agreements created by NASA and valid in outer space, with Australia, Ukraine, and Colombia as signatories. Outer space is used for spaceflight, which is the opposite of astronomical objects.The Artemis Accords are a set of agreements created by NASA and valid in outer space, with Australia, Ukraine, and Colombia as signatories. Outer space is used for spaceflight, which is the opposite of astronomical objects.",
            "output": "[s] Artemis_Accords [r] creator [o] NASA [e] [s] Artemis_Accords [r] valid in place [o] Outer_space [e] [s] Artemis_Accords [r] signatory [o] Australia [e] [s] Artemis_Accords [r] signatory [o] Ukraine [e] [s] Artemis_Accords [r] signatory [o] Colombia [e] [s] outer_space [r] use [o] Spaceflight [e] [s] Outer_space [r] opposite of [o] Astronomical_object [e] [end]",
        },
        {
            "input": "Papaver rhoeas is a species of plant that is invasive to China and has a variety of uses, such as the production of fruit.",
            "output": "[s] Papaver_rhoeas [r] taxon rank [o] Species [e] [s] Papaver_rhoeas [r] invasive to [o] China [e] [s] Papaver_rhoeas [r] use [o] Fruit [e] [s] Papaver_rhoeas [r] instance of [o] Taxon [e] [end]",
        },
        {
            "input": "Gymnastics at the 2006 Commonwealth Games was a sport held in Australia as part of the 2006 Commonwealth Games, an instance of Season (sports). Australia has a diplomatic relation with Mauritius.",
            "output": "[s] Gymnastics_at_the_2006_Commonwealth_Games [r] sport [o] Gymnastics [e] [s] Gymnastics_at_the_2006_Commonwealth_Games [r] part of [o] 2006_Commonwealth_Games [e] [s] Gymnastics_at_the_2006_Commonwealth_Games [r] country [o] Australia [e] [s] Gymnastics_at_the_2006_Commonwealth_Games [r] instance of [o] Season_(sports) [e] [s] Australia [r] diplomatic relation [o] Mauritius [e] [end]",
        },
        {
            "input": "Archibald Campbell, 2nd Earl of Argyll was a noble title held by Archibald Campbell, according to Nordisk familjebok. He belonged to the Clan Campbell family.",
            "output": "[s] Archibald_Campbell,_2nd_Earl_of_Argyll [r] noble title [o] Earl [e] [s] Archibald_Campbell,_2nd_Earl_of_Argyll [r] described by source [o] Nordisk_familjebok [e] [s] Archibald_Campbell,_2nd_Earl_of_Argyll [r] family [o] Clan_Campbell [e] [end]",
        },
        {
            "input": "Arena Zagreb is occupied by RK Zagreb and owned by Zagreb. It hosts a variety of sports, including tennis, and was the site of a groundbreaking event.",
            "output": "[s] Arena_Zagreb [r] owned by [o] Zagreb [e] [s] Arena_Zagreb [r] occupant [o] RK_Zagreb [e] [s] Arena_Zagreb [r] sport [o] Tennis [e] [s] Arena_Zagreb [r] significant event [o] groundbreaking [e] [end]",
        },
        {
            "input": "An accelerometer is a measuring instrument that measures acceleration, which is a physical quantity that follows velocity. Pierre Varignon is credited as the discoverer or inventor of acceleration. The Brockhaus and Efron Encyclopedic Dictionary describes acceleration.",
            "output": "[s] Accelerometer [r] subclass of [o] Measuring_instrument [e] [s] Accelerometer [r] measures [o] Acceleration [e] [s] Acceleration [r] subclass of [o] Physical_quantity [e] [s] Acceleration [r] follows [o] Velocity [e] [s] Acceleration [r] discoverer or inventor [o] Pierre_Varignon [e] [s] Acceleration [r] described by source [o] Brockhaus_and_Efron_Encyclopedic_Dictionary [e] [end]",
        },
        {
            "input": "Kensington and Chelsea Tenant Management Organisation (Kensington and Chelsea TMO) is a Limited company, located in Kensington, with Emma Dent Coad as one of its board members.",
            "output": "[s] Kensington_and_Chelsea_TMO [r] located in the administrative territorial entity [o] Kensington [e] [s] Kensington_and_Chelsea_TMO [r] legal form [o] Limited_company [e] [s] Kensington_and_Chelsea_TMO [r] board member [o] Emma_Dent_Coad [e] [end]",
        },
    ]

    # runtime just has 'input' key
    runtime_input = {"input": "What is the capital of France?"}

    # Check if random selection is working
    synthie_fe_prompter = BasePrompter(
        context="",
        task="",
        instruction="Extract the triples in fully-expanded format from texts below.",
        demo_pool=demo_pool,
        num_demo=4,
        seed=42,
        demo_selection="first",
        output_prefix=' """',
    )

    synthie_fe_prompter.pretty_print()

    prompt = synthie_fe_prompter.materialize(runtime_input)
    print(prompt)

    file_path = os.path.join(PROMPTER_DIR, "IE", "synthie_fe.json")

    synthie_fe_prompter.save(filename=file_path)

    BasePrompter.from_local(file_path).pretty_print()
