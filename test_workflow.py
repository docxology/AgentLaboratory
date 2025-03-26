from agent_lab.visualization.pdf_generator import generate_latex_document; from agent_lab.core.workflow import LaboratoryWorkflow; import yaml, os; config_path = os.path.join("configs", "sample", "POMDP_ActiveInference.yaml"); with open(config_path, "r") as f: config = yaml.safe_load(f); workflow = LaboratoryWorkflow(config); print("Successfully initialized workflow with POMDP config")
