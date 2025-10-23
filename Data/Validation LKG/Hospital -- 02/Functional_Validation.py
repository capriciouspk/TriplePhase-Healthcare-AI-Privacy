from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

# 1. Load your knowledge graph
kg = Graph()
kg.parse("hospital_1_LKG.ttl", format="turtle")

# 2. Prepare your SPARQL query with proper namespace
query = prepareQuery("""
    SELECT ?patient WHERE {
        ?patient a clin:Patient ;
                 clin:hasDiagnosis clin:HeartDisease ;
                 clin:hasFeature clin:age_MiddleAged .
    }
""", initNs={"clin": "http://clinicalkg.org/ontology#"})

# 3. Execute the query
results = kg.query(query)

# 4. Process the results
print(f"Found {len(results)} middle-aged patients with heart disease:")
for row in results:
    print(row.patient)
