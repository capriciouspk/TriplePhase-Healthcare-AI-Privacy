import pandas as pd
import numpy as np
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD, OWL
import pickle
import json
from sklearn.decomposition import PCA

# ----------------------------
# STEP 1: Data Preprocessing
# ----------------------------
print("Step 1: Preprocessing data...")

df = pd.read_csv("softlabels_noised.csv")

# Clinical thresholds (based on medical guidelines)
RANGES = {
    'age': {'Young': (0,40), 'MiddleAged': (40,60), 'Senior': (60,120)},
    'chol': {'Normal': (0,200), 'Borderline': (200,240), 'High': (240,1000)},
    'thalach': {'Low': (0,150), 'Normal': (150,190), 'High': (190,300)},
    'oldpeak': {'Normal': (0,1), 'Mild': (1,2), 'Severe': (2,10)}
}

def map_to_range(value, feature):
    """Convert numerical values to clinical categories"""
    if feature not in RANGES or pd.isna(value):
        return "Unknown"
    for cat, (low, high) in RANGES[feature].items():
        if low <= value <= high:
            return cat
    return "Unknown"

# Save preprocessed data
df_preprocessed = df.copy()
for feature in RANGES.keys():
    df_preprocessed[f"{feature}_category"] = df[feature].apply(
        lambda x: map_to_range(x, feature))
    
df_preprocessed.to_csv("hospital_1_LKG_preprocessed.csv", index=False)

# ----------------------------
# STEP 2: KG Construction
# ----------------------------
print("Step 2: Building knowledge graph...")

kg = Graph()
clin = Namespace("http://clinicalkg.org/ontology#")
kg.bind("clin", clin)

# Define ontology
kg.add((clin.Patient, RDF.type, OWL.Class))
kg.add((clin.Diagnosis, RDF.type, OWL.Class))
kg.add((clin.ClinicalFeature, RDF.type, OWL.Class))
kg.add((clin.hasFeature, RDF.type, OWL.ObjectProperty))
kg.add((clin.hasDiagnosis, RDF.type, OWL.ObjectProperty))
kg.add((clin.hasConfidence, RDF.type, OWL.DatatypeProperty))

for idx, row in df_preprocessed.iterrows():
    patient = clin[f"patient_{idx}"]
    kg.add((patient, RDF.type, clin.Patient))
    
    # Add demographic features
    kg.add((patient, clin.hasSex, Literal("Male" if row['sex'] == 1 else "Female")))
    
    # Add clinical features as categories
    for feature in RANGES.keys():
        category = map_to_range(row[feature], feature)
        feature_node = clin[f"{feature}_{category}"]
        kg.add((feature_node, RDF.type, clin.ClinicalFeature))
        kg.add((patient, clin.hasFeature, feature_node))
    
    # Add diagnosis with confidence
    diagnosis = clin.HeartDisease if row['soft_label_1'] > 0.5 else clin.Healthy
    kg.add((patient, clin.hasDiagnosis, diagnosis))
    kg.add((diagnosis, clin.hasConfidence, Literal(float(row['soft_label_1']), datatype=XSD.float)))

# Save KG in multiple formats
kg.serialize("hospital_1_LKG.ttl", format="turtle")
kg.serialize("hospital_1_LKG.rdf", format="xml")
with open("hospital_1_LKG.pkl", "wb") as f:
    pickle.dump(kg, f)

# ----------------------------
# STEP 3: Safe Embedding Generation
# ----------------------------
print("Step 3: Generating embeddings with safe PCA...")

# Get numerical features (exclude missing values)
numerical_features = df[['age', 'chol', 'thalach', 'oldpeak']].fillna(0)

# Dynamic PCA components (max = min(n_samples, n_features))
n_components = min(3, numerical_features.shape[1], numerical_features.shape[0])
print(f"Using {n_components} PCA components")

pca = PCA(n_components=n_components)
num_embeddings = pca.fit_transform(numerical_features)

# Generate manual KG embeddings
def manual_kg_embeddings(kg, dim=50):
    """Create simple embeddings for KG entities"""
    entities = sorted({str(ent) for ent in kg.subjects() if isinstance(ent, URIRef)} | 
                     {str(ent) for ent in kg.objects() if isinstance(ent, URIRef)})
    return {ent: [float(x) for x in np.random.normal(size=dim)] for ent in entities}

# Convert all numpy arrays to lists for JSON serialization
embeddings = {
    "manual_kg_embeddings": manual_kg_embeddings(kg),
    "numeric_pca_embeddings": num_embeddings.tolist(),
    "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_],
    "feature_names": numerical_features.columns.tolist()
}

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Save embeddings with proper serialization
with open("hospital_1_LKG_embeddings.json", "w") as f:
    json.dump(embeddings, f, cls=NumpyEncoder)

print("Processing complete! Files saved:")
print("- hospital_1_LKG_preprocessed.csv")
print("- hospital_1_LKG.ttl (Turtle format)")
print("- hospital_1_LKG.rdf (RDF/XML)")
print("- hospital_1_LKG.pkl (Pickled graph)")
print("- hospital_1_LKG_embeddings.json")
