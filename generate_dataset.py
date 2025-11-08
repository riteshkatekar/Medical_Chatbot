#!/usr/bin/env python3
"""
generate_dataset.py

Create synthetic, large dataset for your medical chatbot.

Outputs (default):
  - data/dataset.csv
  - data/symptom_description.csv
  - data/symptom_precaution.csv

Configuration:
  - NUM_ROWS controls approximate number of dataset rows generated (default 20000).
  - PER_DISEASE controls minimum rows per disease (the script will balance).
"""

import os
import csv
import random
from itertools import combinations, chain

random.seed(42)

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# --- canonical disease list + core symptom tokens (base set from your descriptions) ---
DISEASE_CORE = {
    "Drug Reaction": ["drug_reaction", "rash", "itching", "swelling", "nausea", "fever"],
    "Malaria": ["fever", "chills", "headache", "sweating", "nausea", "body_pain"],
    "Allergy": ["sneezing", "itching", "runny_nose", "watery_eyes", "rash"],
    "Hypothyroidism": ["fatigue", "weight_gain", "cold_intolerance", "dry_skin", "constipation"],
    "Psoriasis": ["skin_rash", "red_patches", "scaly_skin", "itching"],
    "GERD": ["heartburn", "acid_reflux", "regurgitation", "chest_discomfort"],
    "Chronic cholestasis": ["itching", "jaundice", "dark_urine", "pale_stools"],
    "hepatitis A": ["jaundice", "nausea", "vomiting", "abdominal_pain", "fever"],
    "Osteoarthristis": ["joint_pain", "stiffness", "reduced_range_of_motion"],
    "(vertigo) Paroymsal  Positional Vertigo": ["vertigo", "dizziness", "loss_of_balance"],
    "Hypoglycemia": ["sweating", "palpitations", "tremor", "confusion"],
    "Acne": ["pimples", "pus_filled_pimples", "skin_rash"],
    "Diabetes": ["polyuria", "polydipsia", "polyphagia", "fatigue"],
    "Impetigo": ["skin_rash", "blister", "yellow_crust_ooze", "red_sore_around_nose"],
    "Hypertension": ["headache", "dizziness", "blurred_vision"],
    "Peptic ulcer diseae": ["stomach_pain", "acidity", "heartburn", "nausea"],
    "Dimorphic hemmorhoids(piles)": ["bleeding_per_rectum", "pain_on_defecation", "itching"],
    "Common Cold": ["runny_nose", "sore_throat", "cough", "sneezing"],
    "Chicken pox": ["fever", "blister", "itching", "rash"],
    "Cervical spondylosis": ["neck_pain", "stiff_neck", "shoulder_pain"],
    "Hyperthyroidism": ["weight_loss", "palpitations", "heat_intolerance", "tremor"],
    "Urinary tract infection": ["burning_micturition", "frequent_urination", "lower_abdominal_pain"],
    "Varicose veins": ["leg_pain", "swelling", "visible_veins", "aching"],
    "AIDS": ["weight_loss", "fever", "chronic_infections", "fatigue"],
    "Paralysis (brain hemorrhage)": ["sudden_weakness", "speech_difficulty", "facial_droop"],
    "Typhoid": ["fever", "headache", "constipation", "malaise"],
    "Hepatitis B": ["jaundice", "abdominal_pain", "dark_urine", "fatigue"],
    "Fungal infection": ["itching", "skin_rash", "white_discharge"],
    "Hepatitis C": ["jaundice", "fatigue", "abdominal_pain"],
    "Migraine": ["throbbing_headache", "nausea", "sensitivity_to_light"],
    "Bronchial Asthma": ["wheezing", "shortness_of_breath", "cough"],
    "Alcoholic hepatitis": ["jaundice", "abdominal_pain", "nausea", "history_of_alcohol"],
    "Jaundice": ["jaundice", "yellow_eyes", "dark_urine"],
    "Hepatitis E": ["jaundice", "fever", "abdominal_pain"],
    "Dengue": ["high_fever", "headache", "joint_pain", "rash"],
    "Hepatitis D": ["jaundice", "abdominal_pain", "dark_urine"],
    "Heart attack": ["chest_pain", "shortness_of_breath", "sweating"],
    "Pneumonia": ["fever", "cough", "shortness_of_breath", "chest_pain"],
    "Arthritis": ["joint_pain", "swelling", "stiffness"],
    "Gastroenteritis": ["diarrhoea", "abdominal_pain", "nausea", "vomiting", "fever"],
    "Tuberculosis": ["chronic_cough", "weight_loss", "fever", "night_sweats"],
    # Add the 10 common day-to-day 'symptom labels' as standalone entries:
    "Headache": ["headache"],
    "Fever": ["fever", "high_temperature"],
    "Cough": ["cough"],
    "Cold / Runny nose": ["runny_nose", "sneezing", "nasal_congestion"],
    "Sore throat": ["sore_throat", "throat_pain"],
    "Stomach pain": ["stomach_pain", "abdominal_pain"],
    "Vomiting": ["vomiting", "nausea"],
    "Diarrhoea": ["diarrhoea"],
    "Body pain": ["body_pain", "myalgia"],
    "Fatigue": ["fatigue", "tiredness"],
}

# flavor tokens that will be used to create variety
ADJECTIVES = ["mild", "moderate", "severe", "intermittent", "persistent", "sudden"]
CONNECTORS = ["and", "with", "plus"]

NUM_ROWS = 20000  # default size; change as needed

# --- helper to create a symptom list string formatted like your dataset rows (columns after disease) ---
def synth_symptom_row(core):
    # choose random 1..5 core symptoms + occasional adjective
    k = max(1, min(len(core), random.choices([1,2,3,4,5], [0.1,0.2,0.4,0.2,0.1])[0]))
    chosen = random.sample(core, k)
    # sometimes add adjectives to 30% of symptoms
    def decorate(tok):
        if random.random() < 0.28:
            adj = random.choice(ADJECTIVES)
            return f"{adj}_{tok}"
        return tok
    chosen = [decorate(c) for c in chosen]
    # shuffle
    random.shuffle(chosen)
    # ensure tokens have underscores (consistent with your dataset)
    return chosen

# Build distribution: ensure minimum per-disease then add extras
disease_list = list(DISEASE_CORE.keys())
rows_per_disease = {d: max(50, NUM_ROWS // len(disease_list)) for d in disease_list}  # base
# adjust to reach NUM_ROWS
current = sum(rows_per_disease.values())
i = 0
while current < NUM_ROWS:
    rows_per_disease[disease_list[i % len(disease_list)]] += 1
    current += 1
    i += 1

# Write dataset.csv: disease, symptom1, symptom2, ...
dataset_path = os.path.join(OUT_DIR, "dataset.csv")
with open(dataset_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for disease, nrows in rows_per_disease.items():
        core = DISEASE_CORE[disease]
        for _ in range(nrows):
            syms = synth_symptom_row(core)
            row = [disease] + syms
            writer.writerow(row)

print(f"Wrote dataset: {dataset_path}  (rows approx: {NUM_ROWS})")

# Write symptom_description.csv (two-column)
desc_path = os.path.join(OUT_DIR, "symptom_description.csv")
DESCRIPTIONS = {
    # Use your given long descriptions for many keys (trimmed here to keep file readable)
    # I'll fill from your provided `system_description` — copy/paste long text where required.
    "Drug Reaction": "An adverse drug reaction (ADR) is an injury caused by taking medication. ADRs may occur following a single dose or prolonged administration.",
    "Malaria": "An infectious disease caused by protozoan parasites from the Plasmodium family that can be transmitted by the bite of the Anopheles mosquito. Falciparum malaria is the most severe.",
    "Allergy": "An allergy is the immune system responding to a normally harmless foreign substance such as pollen, food, or pet dander.",
    "Hypothyroidism": "Hypothyroidism is an underactive thyroid causing fatigue, weight gain and cold intolerance.",
    "Psoriasis": "Psoriasis is a common skin disorder forming thick, red scaly patches often on scalp or joints.",
    "GERD": "Gastroesophageal reflux disease (GERD) causes reflux/heartburn due to lower esophageal sphincter dysfunction.",
    "Chronic cholestasis": "Chronic cholestatic diseases are characterized by defective bile acid transport from liver to intestine.",
    "hepatitis A": "Hepatitis A is a contagious liver infection caused by the hepatitis A virus.",
    "Osteoarthristis": "Osteoarthritis is wear-and-tear arthritis causing joint pain and stiffness.",
    "(vertigo) Paroymsal  Positional Vertigo": "Benign paroxysmal positional vertigo (BPPV) causes brief episodes of dizziness triggered by head movements.",
    "Hypoglycemia": "Hypoglycemia is low blood sugar causing sweating, tremor, confusion; often related to diabetes treatment.",
    "Acne": "Acne vulgaris is formation of pimples and pustules in pilosebaceous units.",
    "Diabetes": "Diabetes is high blood glucose; symptoms may include increased thirst, urination and fatigue.",
    "Impetigo": "Impetigo is a contagious skin infection producing honey-colored crusted sores.",
    "Hypertension": "Hypertension is persistent high blood pressure, often asymptomatic.",
    "Peptic ulcer diseae": "Peptic ulcer disease is a sore in stomach or first part of small intestine causing pain and bleeding.",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoidal disease (piles) are swollen vascular structures in the anal canal causing pain/bleeding.",
    "Common Cold": "The common cold is a viral infection of the upper respiratory tract causing runny nose, cough, sore throat.",
    "Chicken pox": "Chickenpox is a contagious viral illness causing itchy blisters and fever.",
    "Cervical spondylosis": "Cervical spondylosis is age-related wear affecting neck spinal disks and causing neck pain.",
    "Hyperthyroidism": "Hyperthyroidism is excessive thyroid hormone causing weight loss, palpitations, heat intolerance.",
    "Urinary tract infection": "UTI is infection of urinary tract; common symptoms are burning urination and frequent urge.",
    "Varicose veins": "Enlarged twisted veins, usually in legs, causing aching and swelling.",
    "AIDS": "Acquired immunodeficiency syndrome (AIDS) is caused by HIV and leads to immune dysfunction.",
    "Paralysis (brain hemorrhage)": "Intracerebral hemorrhage causes sudden neurological deficits such as weakness or paralysis.",
    "Typhoid": "Typhoid fever is caused by Salmonella typhi and produces prolonged fever, headache and malaise.",
    "Hepatitis B": "Hepatitis B causes liver inflammation; can lead to chronic disease and cirrhosis.",
    "Fungal infection": "Fungal infections affect skin or mucosa; common symptom is itching and rash.",
    "Hepatitis C": "Hepatitis C is inflammation of liver due to HCV; may progress to chronic liver disease.",
    "Migraine": "Migraine causes throbbing unilateral headache with nausea and light sensitivity.",
    "Bronchial Asthma": "Bronchial asthma is chronic airway inflammation causing wheeze and breathlessness.",
    "Alcoholic hepatitis": "Alcoholic hepatitis is liver inflammation from heavy alcohol use.",
    "Jaundice": "Jaundice is yellowing of skin and eyes due to elevated bilirubin.",
    "Hepatitis E": "Hepatitis E is liver inflammation transmitted by contaminated food/water.",
    "Dengue": "Dengue is mosquito-borne viral illness with high fever, severe joint pain and rash.",
    "Hepatitis D": "Hepatitis D is a delta virus causing liver inflammation usually with HBV co-infection.",
    "Heart attack": "Myocardial infarction (heart attack) is death of heart muscle due to loss of blood supply.",
    "Pneumonia": "Pneumonia is infection of lungs causing cough, fever, and difficulty breathing.",
    "Arthritis": "Arthritis is joint inflammation causing pain and decreased mobility.",
    "Gastroenteritis": "Gastroenteritis is inflammation of digestive tract causing diarrhea, cramps and vomiting.",
    "Tuberculosis": "Tuberculosis (TB) commonly affects lungs causing chronic cough, weight loss and fever.",
    # Common symptom labels
    "Headache": "Headache is pain anywhere in the head; many causes range from tension to migraine.",
    "Fever": "Fever (high temperature) is a systemic sign of infection or inflammation.",
    "Cough": "Cough is a reflex to clear the airway often from infections or irritants.",
    "Cold / Runny nose": "Runny nose and cold symptoms are common upper respiratory infections.",
    "Sore throat": "Sore throat is pain or irritation of the throat commonly from infection or inflammation.",
    "Stomach pain": "Stomach pain refers to abdominal discomfort from many causes such as gastritis or infection.",
    "Vomiting": "Vomiting is forceful expulsion of stomach contents; sign of GI irritation or infection.",
    "Diarrhoea": "Diarrhoea is loose or watery stools often due to infection, food poisoning, or intolerance.",
    "Body pain": "Body pain (myalgia) is generalized muscle aches often seen in viral infections or strain.",
    "Fatigue": "Fatigue is extreme tiredness that can result from many medical or lifestyle causes."
}

with open(desc_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for k, v in DESCRIPTIONS.items():
        writer.writerow([k, v])
print(f"Wrote descriptions: {desc_path}")

# Write symptom_precaution.csv
prec_path = os.path.join(OUT_DIR, "symptom_precaution.csv")
PRECAUTIONS = {
    # Use your provided precaution text where appropriate, else provide safe generic ones
    "Drug Reaction": ["stop irritation", "consult nearest hospital", "stop taking drug", "follow up"],
    "Malaria": ["Consult nearest hospital", "avoid oily food", "avoid non veg food", "keep mosquitos out"],
    "Allergy": ["apply calamine", "cover area with bandage", "use ice to compress itching"],
    "Hypothyroidism": ["reduce stress", "exercise", "eat healthy", "get proper sleep"],
    "Psoriasis": ["wash hands with warm soapy water", "stop bleeding using pressure", "consult doctor", "salt baths"],
    "GERD": ["avoid fatty spicy food", "avoid lying down after eating", "maintain healthy weight", "exercise"],
    "Chronic cholestasis":["cold baths","anti itch medicine","consult doctor","eat healthy"],
    "hepatitis A":["Consult nearest hospital","wash hands through","avoid fatty spicy food","medication"],
    "Osteoarthristis":["acetaminophen","consult nearest hospital","follow up","salt baths"],
    "(vertigo) Paroymsal  Positional Vertigo":["lie down","avoid sudden change in body","avoid abrupt head movment","relax"],
    "Hypoglycemia":["lie down on side","check in pulse","drink sugary drinks","consult doctor"],
    "Acne":["bath twice","avoid fatty spicy food","drink plenty of water","avoid too many products"],
    "Diabetes":["have balanced diet","exercise","consult doctor","follow up"],
    "Impetigo":["soak affected area in warm water","use antibiotics","remove scabs with wet compressed cloth","consult doctor"],
    "Hypertension":["meditation","salt baths","reduce stress","get proper sleep"],
    "Peptic ulcer diseae":["avoid fatty spicy food","consume probiotic food","eliminate milk","limit alcohol"],
    "Dimorphic hemmorhoids(piles)":["avoid fatty spicy food","consume witch hazel","warm bath with epsom salt","consume alovera juice"],
    "Common Cold":["drink vitamin c rich drinks","take vapour","avoid cold food","keep fever in check"],
    "Chicken pox":["use neem in bathing","consume neem leaves","take vaccine","avoid public places"],
    "Cervical spondylosis":["use heating pad or cold pack","exercise","take otc pain reliver","consult doctor"],
    "Hyperthyroidism":["eat healthy","massage","use lemon balm","take radioactive iodine treatment"],
    "Urinary tract infection":["drink plenty of water","increase vitamin c intake","drink cranberry juice","take probiotics"],
    "Varicose veins":["lie down flat and raise the leg high","use oinments","use vein compression","dont stand still for long"],
    "AIDS":["avoid open cuts","wear ppe if possible","consult doctor","follow up"],
    "Paralysis (brain hemorrhage)":["massage","eat healthy","exercise","consult doctor"],
    "Typhoid":["eat high calorie vegitables","antiboitic therapy","consult doctor","medication"],
    "Hepatitis B":["consult nearest hospital","vaccination","eat healthy","medication"],
    "Fungal infection":["bath twice","use detol or neem in bathing water","keep infected area dry","use clean cloths"],
    "Hepatitis C":["Consult nearest hospital","vaccination","eat healthy","medication"],
    "Migraine":["meditation","reduce stress","use poloroid glasses in sun","consult doctor"],
    "Bronchial Asthma":["switch to loose cloothing","take deep breaths","get away from trigger","seek help"],
    "Alcoholic hepatitis":["stop alcohol consumption","consult doctor","medication","follow up"],
    "Jaundice":["drink plenty of water","consume milk thistle","eat fruits and high fiberous food","medication"],
    "Hepatitis E":["stop alcohol consumption","rest","consult doctor","medication"],
    "Dengue":["drink papaya leaf juice","avoid fatty spicy food","keep mosquitos away","keep hydrated"],
    "Hepatitis D":["consult doctor","medication","eat healthy","follow up"],
    "Heart attack":["call ambulance","chew or swallow asprin","keep calm"],
    "Pneumonia":["consult doctor","medication","rest","follow up"],
    "Arthritis":["exercise","use hot and cold therapy","try acupuncture","massage"],
    "Gastroenteritis":["stop eating solid food for while","try taking small sips of water","rest","ease back into eating"],
    "Tuberculosis":["cover mouth","consult doctor","medication","rest"],
    # Common symptom labels: give safe generic precautions
    "Headache":["rest","drink water","avoid bright lights","use cold compress"],
    "Fever":["stay hydrated","rest","monitor temperature","seek care if very high or prolonged"],
    "Cough":["stay hydrated","use steam inhalation","avoid smoke","see doctor if breathless"],
    "Cold / Runny nose":["rest","use saline nasal drops","stay warm","avoid cold food"],
    "Sore throat":["gargle with saline","stay hydrated","lozenges for comfort","see doctor if severe"],
    "Stomach pain":["stop solid food briefly","sip oral rehydration","avoid spicy food","consult doctor if severe"],
    "Vomiting":["sip small amounts of clear fluids","avoid solid food until vomiting stops","seek care if unable to retain fluids"],
    "Diarrhoea":["oral rehydration","avoid dairy for short period","eat bland foods","seek care if dehydrated"],
    "Body pain":["rest","paracetamol if needed (follow medical advice)","stay hydrated","light stretching"],
    "Fatigue":["rest","sleep hygiene","balanced diet","consult if prolonged"]
}

with open(prec_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for k, lst in PRECAUTIONS.items():
        writer.writerow([k] + lst)

print(f"Wrote precautions: {prec_path}")

print("Generation complete.")
