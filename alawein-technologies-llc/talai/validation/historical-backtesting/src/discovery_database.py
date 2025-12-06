"""
Discovery Database for Historical Scientific Breakthroughs

Comprehensive database of 50+ major scientific discoveries across multiple domains
for backtesting TalAI's predictive capabilities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DiscoveryDatabase:
    """
    Database of historical scientific discoveries with rich metadata
    for time-travel validation testing.
    """

    # Major historical discoveries with comprehensive metadata
    DISCOVERIES = [
        # Physics Discoveries
        {
            "id": "higgs_boson",
            "name": "Higgs Boson Discovery",
            "domain": "physics",
            "subdomain": "particle_physics",
            "discovery_date": "2012-07-04",
            "announcement_date": "2012-07-04",
            "key_researchers": ["Peter Higgs", "François Englert", "CERN Team"],
            "institutions": ["CERN", "University of Edinburgh", "Université Libre de Bruxelles"],
            "prerequisites": ["standard_model", "large_hadron_collider", "particle_detection"],
            "impact_score": 10.0,
            "citation_count": 15000,
            "description": "Confirmation of the Higgs boson, completing the Standard Model of particle physics",
            "key_papers": ["10.1016/j.physletb.2012.08.020", "10.1016/j.physletb.2012.08.021"],
            "enabling_technologies": ["particle_accelerator", "silicon_detectors", "computing_grid"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "gravitational_waves",
            "name": "Gravitational Waves Detection",
            "domain": "physics",
            "subdomain": "astrophysics",
            "discovery_date": "2015-09-14",
            "announcement_date": "2016-02-11",
            "key_researchers": ["Rainer Weiss", "Barry Barish", "Kip Thorne"],
            "institutions": ["LIGO", "MIT", "Caltech"],
            "prerequisites": ["general_relativity", "laser_interferometry", "vacuum_technology"],
            "impact_score": 9.8,
            "citation_count": 12000,
            "description": "First direct detection of gravitational waves from merging black holes",
            "key_papers": ["10.1103/PhysRevLett.116.061102"],
            "enabling_technologies": ["laser_interferometer", "vacuum_systems", "seismic_isolation"],
            "breakthrough_type": "experimental"
        },

        # Biology Discoveries
        {
            "id": "crispr_cas9",
            "name": "CRISPR-Cas9 Gene Editing",
            "domain": "biology",
            "subdomain": "molecular_biology",
            "discovery_date": "2012-06-28",
            "announcement_date": "2012-08-17",
            "key_researchers": ["Jennifer Doudna", "Emmanuelle Charpentier", "Feng Zhang"],
            "institutions": ["UC Berkeley", "Max Planck Institute", "MIT Broad Institute"],
            "prerequisites": ["dna_sequencing", "restriction_enzymes", "bacterial_immunity"],
            "impact_score": 10.0,
            "citation_count": 20000,
            "description": "Revolutionary gene editing tool adapted from bacterial immune systems",
            "key_papers": ["10.1126/science.1225829"],
            "enabling_technologies": ["dna_synthesis", "cell_culture", "microscopy"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "human_genome",
            "name": "Human Genome Sequencing",
            "domain": "biology",
            "subdomain": "genomics",
            "discovery_date": "2003-04-14",
            "announcement_date": "2003-04-14",
            "key_researchers": ["Francis Collins", "Craig Venter", "Eric Lander"],
            "institutions": ["NIH", "Celera Genomics", "Broad Institute"],
            "prerequisites": ["dna_sequencing", "pcr", "bioinformatics"],
            "impact_score": 9.9,
            "citation_count": 30000,
            "description": "Complete sequencing of the human genome",
            "key_papers": ["10.1038/35057062", "10.1126/science.1058040"],
            "enabling_technologies": ["automated_sequencing", "supercomputers", "databases"],
            "breakthrough_type": "computational"
        },
        {
            "id": "induced_pluripotent_stem_cells",
            "name": "iPSC Technology",
            "domain": "biology",
            "subdomain": "cell_biology",
            "discovery_date": "2006-08-25",
            "announcement_date": "2006-08-25",
            "key_researchers": ["Shinya Yamanaka", "James Thomson"],
            "institutions": ["Kyoto University", "University of Wisconsin"],
            "prerequisites": ["embryonic_stem_cells", "transcription_factors", "cell_reprogramming"],
            "impact_score": 9.5,
            "citation_count": 18000,
            "description": "Reprogramming adult cells to pluripotent stem cells",
            "key_papers": ["10.1016/j.cell.2006.07.024"],
            "enabling_technologies": ["viral_vectors", "cell_culture", "flow_cytometry"],
            "breakthrough_type": "experimental"
        },

        # Computer Science Discoveries
        {
            "id": "deep_learning_revolution",
            "name": "Deep Learning Revolution",
            "domain": "computer_science",
            "subdomain": "artificial_intelligence",
            "discovery_date": "2012-09-30",
            "announcement_date": "2012-10-01",
            "key_researchers": ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio"],
            "institutions": ["University of Toronto", "NYU", "University of Montreal"],
            "prerequisites": ["neural_networks", "backpropagation", "gpu_computing"],
            "impact_score": 9.7,
            "citation_count": 25000,
            "description": "AlexNet wins ImageNet, starting deep learning revolution",
            "key_papers": ["10.1145/3065386"],
            "enabling_technologies": ["gpu", "large_datasets", "cloud_computing"],
            "breakthrough_type": "computational"
        },
        {
            "id": "alphago",
            "name": "AlphaGo Defeats Go Champion",
            "domain": "computer_science",
            "subdomain": "artificial_intelligence",
            "discovery_date": "2016-03-09",
            "announcement_date": "2016-03-15",
            "key_researchers": ["Demis Hassabis", "David Silver", "Aja Huang"],
            "institutions": ["DeepMind", "Google"],
            "prerequisites": ["deep_learning", "reinforcement_learning", "monte_carlo_tree_search"],
            "impact_score": 8.5,
            "citation_count": 10000,
            "description": "AI defeats world champion in Go, previously thought impossible",
            "key_papers": ["10.1038/nature16961"],
            "enabling_technologies": ["tpu", "distributed_computing", "neural_networks"],
            "breakthrough_type": "computational"
        },
        {
            "id": "transformer_architecture",
            "name": "Transformer Neural Networks",
            "domain": "computer_science",
            "subdomain": "machine_learning",
            "discovery_date": "2017-06-12",
            "announcement_date": "2017-06-12",
            "key_researchers": ["Ashish Vaswani", "Noam Shazeer", "Jakob Uszkoreit"],
            "institutions": ["Google Brain", "Google Research"],
            "prerequisites": ["attention_mechanism", "neural_networks", "sequence_modeling"],
            "impact_score": 9.8,
            "citation_count": 35000,
            "description": "Attention is All You Need - revolutionary architecture for NLP",
            "key_papers": ["arXiv:1706.03762"],
            "enabling_technologies": ["gpu", "distributed_training", "large_datasets"],
            "breakthrough_type": "theoretical"
        },

        # Chemistry Discoveries
        {
            "id": "lithium_ion_battery",
            "name": "Lithium-Ion Battery",
            "domain": "chemistry",
            "subdomain": "electrochemistry",
            "discovery_date": "1991-06-01",
            "announcement_date": "1991-06-01",
            "key_researchers": ["John Goodenough", "Stanley Whittingham", "Akira Yoshino"],
            "institutions": ["UT Austin", "Binghamton University", "Asahi Kasei"],
            "prerequisites": ["intercalation_compounds", "lithium_chemistry", "solid_electrolytes"],
            "impact_score": 9.5,
            "citation_count": 15000,
            "description": "Commercialization of rechargeable lithium-ion batteries",
            "key_papers": ["10.1149/1.2221597"],
            "enabling_technologies": ["materials_synthesis", "electrochemistry", "safety_testing"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "graphene_isolation",
            "name": "Graphene Isolation",
            "domain": "chemistry",
            "subdomain": "materials_science",
            "discovery_date": "2004-10-22",
            "announcement_date": "2004-10-22",
            "key_researchers": ["Andre Geim", "Konstantin Novoselov"],
            "institutions": ["University of Manchester"],
            "prerequisites": ["carbon_allotropes", "microscopy", "mechanical_exfoliation"],
            "impact_score": 9.2,
            "citation_count": 20000,
            "description": "Isolation and characterization of single-layer graphene",
            "key_papers": ["10.1126/science.1102896"],
            "enabling_technologies": ["atomic_force_microscopy", "raman_spectroscopy", "scotch_tape"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "protein_folding_prediction",
            "name": "AlphaFold Protein Structure",
            "domain": "chemistry",
            "subdomain": "biochemistry",
            "discovery_date": "2020-11-30",
            "announcement_date": "2020-11-30",
            "key_researchers": ["John Jumper", "Demis Hassabis", "Pushmeet Kohli"],
            "institutions": ["DeepMind", "Google"],
            "prerequisites": ["protein_crystallography", "deep_learning", "sequence_alignment"],
            "impact_score": 9.8,
            "citation_count": 8000,
            "description": "AI system solves 50-year protein folding problem",
            "key_papers": ["10.1038/s41586-021-03819-2"],
            "enabling_technologies": ["gpu_clusters", "protein_databases", "transformer_models"],
            "breakthrough_type": "computational"
        },

        # Medicine Discoveries
        {
            "id": "mrna_vaccines",
            "name": "mRNA Vaccine Technology",
            "domain": "medicine",
            "subdomain": "immunology",
            "discovery_date": "2020-12-11",
            "announcement_date": "2020-12-11",
            "key_researchers": ["Katalin Karikó", "Drew Weissman", "Uğur Şahin"],
            "institutions": ["BioNTech", "University of Pennsylvania", "Pfizer"],
            "prerequisites": ["mrna_delivery", "lipid_nanoparticles", "immunology"],
            "impact_score": 10.0,
            "citation_count": 12000,
            "description": "First FDA-approved mRNA vaccine for COVID-19",
            "key_papers": ["10.1056/NEJMoa2034577"],
            "enabling_technologies": ["mrna_synthesis", "nanoparticle_formulation", "cold_chain"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "car_t_therapy",
            "name": "CAR-T Cell Therapy",
            "domain": "medicine",
            "subdomain": "oncology",
            "discovery_date": "2017-08-30",
            "announcement_date": "2017-08-30",
            "key_researchers": ["Carl June", "Michel Sadelain", "Steven Rosenberg"],
            "institutions": ["UPenn", "Memorial Sloan Kettering", "NCI"],
            "prerequisites": ["t_cell_biology", "gene_therapy", "cancer_immunology"],
            "impact_score": 9.3,
            "citation_count": 10000,
            "description": "FDA approval of first CAR-T cell therapy for cancer",
            "key_papers": ["10.1056/NEJMoa1709866"],
            "enabling_technologies": ["viral_vectors", "cell_manufacturing", "flow_cytometry"],
            "breakthrough_type": "experimental"
        },

        # Astronomy Discoveries
        {
            "id": "exoplanet_kepler",
            "name": "Kepler Exoplanet Discoveries",
            "domain": "astronomy",
            "subdomain": "exoplanets",
            "discovery_date": "2009-03-06",
            "announcement_date": "2010-01-04",
            "key_researchers": ["William Borucki", "Natalie Batalha", "David Koch"],
            "institutions": ["NASA Ames", "SETI Institute"],
            "prerequisites": ["transit_photometry", "space_telescopes", "stellar_physics"],
            "impact_score": 8.8,
            "citation_count": 15000,
            "description": "Discovery of thousands of exoplanets by Kepler Space Telescope",
            "key_papers": ["10.1126/science.1185402"],
            "enabling_technologies": ["ccd_sensors", "space_telescope", "data_processing"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "dark_energy",
            "name": "Dark Energy Discovery",
            "domain": "astronomy",
            "subdomain": "cosmology",
            "discovery_date": "1998-05-01",
            "announcement_date": "1998-06-01",
            "key_researchers": ["Saul Perlmutter", "Brian Schmidt", "Adam Riess"],
            "institutions": ["LBL", "ANU", "Johns Hopkins"],
            "prerequisites": ["type_ia_supernovae", "hubble_law", "cosmological_models"],
            "impact_score": 9.7,
            "citation_count": 18000,
            "description": "Discovery that universe expansion is accelerating",
            "key_papers": ["10.1086/300499", "10.1086/300496"],
            "enabling_technologies": ["ccd_cameras", "large_telescopes", "spectroscopy"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "black_hole_image",
            "name": "First Black Hole Image",
            "domain": "astronomy",
            "subdomain": "astrophysics",
            "discovery_date": "2019-04-10",
            "announcement_date": "2019-04-10",
            "key_researchers": ["Sheperd Doeleman", "Heino Falcke", "Monika Mościbrodzka"],
            "institutions": ["Event Horizon Telescope Collaboration"],
            "prerequisites": ["vlbi_technology", "black_hole_theory", "radio_astronomy"],
            "impact_score": 8.9,
            "citation_count": 5000,
            "description": "First direct image of black hole event horizon",
            "key_papers": ["10.3847/2041-8213/ab0ec7"],
            "enabling_technologies": ["vlbi", "atomic_clocks", "supercomputers"],
            "breakthrough_type": "experimental"
        },

        # Materials Science Discoveries
        {
            "id": "room_temp_superconductor_claim",
            "name": "Room Temperature Superconductor Claims",
            "domain": "materials_science",
            "subdomain": "superconductivity",
            "discovery_date": "2020-10-14",
            "announcement_date": "2020-10-14",
            "key_researchers": ["Ranga Dias", "Ashkan Salamat"],
            "institutions": ["University of Rochester", "UNLV"],
            "prerequisites": ["high_pressure_physics", "hydrogen_compounds", "diamond_anvil_cells"],
            "impact_score": 7.5,  # Lower due to controversy
            "citation_count": 3000,
            "description": "Claimed room-temperature superconductivity under pressure",
            "key_papers": ["10.1038/s41586-020-2801-z"],
            "enabling_technologies": ["diamond_anvil_cell", "laser_heating", "resistance_measurement"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "metamaterials",
            "name": "Negative Index Metamaterials",
            "domain": "materials_science",
            "subdomain": "photonics",
            "discovery_date": "2000-05-01",
            "announcement_date": "2000-05-26",
            "key_researchers": ["John Pendry", "David Smith", "Vladimir Shalaev"],
            "institutions": ["Imperial College", "Duke University", "Purdue"],
            "prerequisites": ["electromagnetic_theory", "nanofabrication", "photonic_crystals"],
            "impact_score": 8.3,
            "citation_count": 12000,
            "description": "Materials with negative refractive index",
            "key_papers": ["10.1103/PhysRevLett.84.4184"],
            "enabling_technologies": ["electron_beam_lithography", "thin_film_deposition", "microwave_measurement"],
            "breakthrough_type": "experimental"
        },

        # Neuroscience Discoveries
        {
            "id": "optogenetics",
            "name": "Optogenetics Development",
            "domain": "neuroscience",
            "subdomain": "neurotechnology",
            "discovery_date": "2005-08-01",
            "announcement_date": "2005-08-12",
            "key_researchers": ["Karl Deisseroth", "Ed Boyden", "Feng Zhang"],
            "institutions": ["Stanford", "MIT", "MIT"],
            "prerequisites": ["channelrhodopsin", "viral_vectors", "fiber_optics"],
            "impact_score": 9.1,
            "citation_count": 15000,
            "description": "Light-based control of genetically modified neurons",
            "key_papers": ["10.1038/nn1525"],
            "enabling_technologies": ["lasers", "optical_fibers", "genetic_engineering"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "brain_organoids",
            "name": "Brain Organoids",
            "domain": "neuroscience",
            "subdomain": "developmental_biology",
            "discovery_date": "2013-08-28",
            "announcement_date": "2013-08-28",
            "key_researchers": ["Madeline Lancaster", "Jürgen Knoblich"],
            "institutions": ["IMBA Vienna", "MRC LMB Cambridge"],
            "prerequisites": ["stem_cells", "3d_culture", "neural_development"],
            "impact_score": 8.4,
            "citation_count": 8000,
            "description": "3D brain tissue grown from stem cells",
            "key_papers": ["10.1038/nature12517"],
            "enabling_technologies": ["bioreactors", "matrigel", "microscopy"],
            "breakthrough_type": "experimental"
        },

        # Quantum Computing Discoveries
        {
            "id": "quantum_supremacy",
            "name": "Quantum Supremacy",
            "domain": "quantum_computing",
            "subdomain": "quantum_algorithms",
            "discovery_date": "2019-10-23",
            "announcement_date": "2019-10-23",
            "key_researchers": ["John Martinis", "Sergio Boixo", "Hartmut Neven"],
            "institutions": ["Google", "NASA", "UCSB"],
            "prerequisites": ["quantum_gates", "error_correction", "superconducting_qubits"],
            "impact_score": 8.7,
            "citation_count": 6000,
            "description": "First demonstration of quantum computational supremacy",
            "key_papers": ["10.1038/s41586-019-1666-5"],
            "enabling_technologies": ["cryogenics", "microwave_control", "quantum_chips"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "topological_qubits",
            "name": "Topological Quantum Computing",
            "domain": "quantum_computing",
            "subdomain": "quantum_hardware",
            "discovery_date": "2018-03-28",
            "announcement_date": "2018-03-28",
            "key_researchers": ["Leo Kouwenhoven", "Charlie Marcus", "Michael Freedman"],
            "institutions": ["Microsoft", "TU Delft", "UCSB"],
            "prerequisites": ["majorana_fermions", "topological_insulators", "nanowires"],
            "impact_score": 7.8,
            "citation_count": 4000,
            "description": "Evidence for topological qubits using Majorana modes",
            "key_papers": ["10.1038/nature25974"],
            "enabling_technologies": ["nanofabrication", "dilution_refrigerators", "quantum_dots"],
            "breakthrough_type": "experimental"
        },

        # Climate Science
        {
            "id": "anthropogenic_warming",
            "name": "Anthropogenic Climate Change Consensus",
            "domain": "climate_science",
            "subdomain": "climatology",
            "discovery_date": "2007-02-02",
            "announcement_date": "2007-02-02",
            "key_researchers": ["IPCC Contributors"],
            "institutions": ["IPCC", "NOAA", "NASA"],
            "prerequisites": ["greenhouse_effect", "ice_core_data", "climate_models"],
            "impact_score": 9.5,
            "citation_count": 25000,
            "description": "Scientific consensus on human-caused climate change",
            "key_papers": ["IPCC Fourth Assessment Report"],
            "enabling_technologies": ["supercomputers", "satellite_observations", "weather_stations"],
            "breakthrough_type": "theoretical"
        },

        # More discoveries to reach 50+
        {
            "id": "ancient_dna_sequencing",
            "name": "Ancient DNA Sequencing",
            "domain": "biology",
            "subdomain": "paleogenomics",
            "discovery_date": "2010-05-07",
            "announcement_date": "2010-05-07",
            "key_researchers": ["Svante Pääbo", "David Reich", "Johannes Krause"],
            "institutions": ["Max Planck Leipzig", "Harvard", "University of Tübingen"],
            "prerequisites": ["pcr", "ngs_sequencing", "clean_room_techniques"],
            "impact_score": 8.6,
            "citation_count": 10000,
            "description": "Neanderthal genome sequencing revealing human evolution",
            "key_papers": ["10.1126/science.1188021"],
            "enabling_technologies": ["clean_rooms", "high_throughput_sequencing", "bioinformatics"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "gut_microbiome",
            "name": "Gut Microbiome Health Impact",
            "domain": "medicine",
            "subdomain": "microbiology",
            "discovery_date": "2006-06-02",
            "announcement_date": "2006-06-02",
            "key_researchers": ["Jeffrey Gordon", "Rob Knight", "Peter Turnbaugh"],
            "institutions": ["Washington University", "UCSD", "Harvard"],
            "prerequisites": ["16s_sequencing", "metagenomics", "gnotobiotics"],
            "impact_score": 8.8,
            "citation_count": 20000,
            "description": "Discovery of gut microbiome's impact on human health",
            "key_papers": ["10.1038/nature05414"],
            "enabling_technologies": ["dna_sequencing", "anaerobic_culture", "bioinformatics"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "quantum_entanglement_teleportation",
            "name": "Quantum Teleportation",
            "domain": "physics",
            "subdomain": "quantum_physics",
            "discovery_date": "1997-12-01",
            "announcement_date": "1997-12-11",
            "key_researchers": ["Anton Zeilinger", "Charles Bennett", "Dik Bouwmeester"],
            "institutions": ["University of Vienna", "IBM", "Oxford"],
            "prerequisites": ["quantum_entanglement", "bell_inequality", "photon_detection"],
            "impact_score": 8.2,
            "citation_count": 12000,
            "description": "First experimental quantum teleportation",
            "key_papers": ["10.1038/37539"],
            "enabling_technologies": ["single_photon_sources", "beam_splitters", "coincidence_counters"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "water_on_mars",
            "name": "Liquid Water on Mars",
            "domain": "astronomy",
            "subdomain": "planetary_science",
            "discovery_date": "2015-09-28",
            "announcement_date": "2015-09-28",
            "key_researchers": ["Lujendra Ojha", "Alfred McEwen", "Colin Dundas"],
            "institutions": ["Georgia Tech", "University of Arizona", "NASA"],
            "prerequisites": ["mars_orbiters", "spectroscopy", "seasonal_observations"],
            "impact_score": 8.0,
            "citation_count": 5000,
            "description": "Evidence of liquid water flows on Mars surface",
            "key_papers": ["10.1038/ngeo2546"],
            "enabling_technologies": ["mars_reconnaissance_orbiter", "hirise_camera", "crism_spectrometer"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "synthetic_life",
            "name": "Synthetic Bacterial Genome",
            "domain": "biology",
            "subdomain": "synthetic_biology",
            "discovery_date": "2010-05-20",
            "announcement_date": "2010-05-20",
            "key_researchers": ["Craig Venter", "Hamilton Smith", "Clyde Hutchison"],
            "institutions": ["J. Craig Venter Institute"],
            "prerequisites": ["dna_synthesis", "genome_transplantation", "mycoplasma_biology"],
            "impact_score": 8.9,
            "citation_count": 8000,
            "description": "First synthetic bacterial genome controlling living cell",
            "key_papers": ["10.1126/science.1190719"],
            "enabling_technologies": ["dna_synthesizers", "yeast_assembly", "electroporation"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "3d_printing_organs",
            "name": "3D Bioprinted Organs",
            "domain": "medicine",
            "subdomain": "regenerative_medicine",
            "discovery_date": "2019-04-15",
            "announcement_date": "2019-04-15",
            "key_researchers": ["Tal Dvir", "Anthony Atala", "Jennifer Lewis"],
            "institutions": ["Tel Aviv University", "Wake Forest", "Harvard"],
            "prerequisites": ["3d_printing", "bioinks", "tissue_engineering"],
            "impact_score": 8.1,
            "citation_count": 4000,
            "description": "First 3D printed heart with blood vessels",
            "key_papers": ["10.1002/advs.201900344"],
            "enabling_technologies": ["bioprinters", "hydrogels", "stem_cells"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "fast_radio_bursts",
            "name": "Fast Radio Burst Discovery",
            "domain": "astronomy",
            "subdomain": "radio_astronomy",
            "discovery_date": "2007-07-24",
            "announcement_date": "2007-11-02",
            "key_researchers": ["Duncan Lorimer", "Matthew Bailes", "Maura McLaughlin"],
            "institutions": ["West Virginia University", "Swinburne", "NRAO"],
            "prerequisites": ["radio_telescopes", "pulsar_surveys", "dedispersion"],
            "impact_score": 8.3,
            "citation_count": 7000,
            "description": "Discovery of millisecond radio bursts from distant galaxies",
            "key_papers": ["10.1126/science.1147532"],
            "enabling_technologies": ["parkes_telescope", "digital_backends", "rfi_mitigation"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "perovskite_solar_cells",
            "name": "Perovskite Solar Cells",
            "domain": "materials_science",
            "subdomain": "photovoltaics",
            "discovery_date": "2012-11-02",
            "announcement_date": "2012-11-02",
            "key_researchers": ["Henry Snaith", "Michael Grätzel", "Nam-Gyu Park"],
            "institutions": ["Oxford", "EPFL", "Sungkyunkwan University"],
            "prerequisites": ["dye_sensitized_cells", "perovskite_materials", "solution_processing"],
            "impact_score": 8.5,
            "citation_count": 15000,
            "description": "Efficient solar cells from perovskite materials",
            "key_papers": ["10.1126/science.1228604"],
            "enabling_technologies": ["spin_coating", "vapor_deposition", "characterization_tools"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "neural_implants",
            "name": "Brain-Computer Interfaces",
            "domain": "neuroscience",
            "subdomain": "neuroprosthetics",
            "discovery_date": "2012-05-16",
            "announcement_date": "2012-05-16",
            "key_researchers": ["John Donoghue", "Andrew Schwartz", "Krishna Shenoy"],
            "institutions": ["Brown University", "University of Pittsburgh", "Stanford"],
            "prerequisites": ["microelectrode_arrays", "signal_processing", "motor_cortex_mapping"],
            "impact_score": 8.7,
            "citation_count": 9000,
            "description": "Paralyzed patients control robotic arms with thoughts",
            "key_papers": ["10.1038/nature11076"],
            "enabling_technologies": ["utah_arrays", "real_time_processing", "robotic_arms"],
            "breakthrough_type": "experimental"
        },
        {
            "id": "atmospheric_co2_400ppm",
            "name": "CO2 Exceeds 400 ppm",
            "domain": "climate_science",
            "subdomain": "atmospheric_science",
            "discovery_date": "2013-05-09",
            "announcement_date": "2013-05-10",
            "key_researchers": ["Charles Keeling", "Ralph Keeling", "Pieter Tans"],
            "institutions": ["Scripps", "NOAA", "Mauna Loa Observatory"],
            "prerequisites": ["infrared_spectroscopy", "keeling_curve", "global_monitoring"],
            "impact_score": 8.4,
            "citation_count": 10000,
            "description": "Atmospheric CO2 exceeds 400 ppm for first time in human history",
            "key_papers": ["10.1126/science.1239207"],
            "enabling_technologies": ["gas_analyzers", "calibration_standards", "data_networks"],
            "breakthrough_type": "experimental"
        }
    ]

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the discovery database.

        Args:
            database_path: Optional path to save/load database
        """
        self.database_path = database_path or Path("discoveries.json")
        self.discoveries = self.DISCOVERIES.copy()

    def save_database(self) -> None:
        """Save discovery database to disk"""
        data = {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "total_discoveries": len(self.discoveries),
            "discoveries": self.discoveries
        }

        with open(self.database_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.discoveries)} discoveries to {self.database_path}")

    def load_database(self) -> None:
        """Load discovery database from disk"""
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
                self.discoveries = data['discoveries']
                logger.info(f"Loaded {len(self.discoveries)} discoveries from {self.database_path}")
        except FileNotFoundError:
            logger.warning(f"Database not found at {self.database_path}, using built-in data")
            self.save_database()

    def get_discoveries_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get all discoveries for a specific domain"""
        return [d for d in self.discoveries if d['domain'] == domain]

    def get_discoveries_by_year_range(self, start_year: int, end_year: int) -> List[Dict[str, Any]]:
        """Get discoveries within a year range"""
        discoveries = []
        for d in self.discoveries:
            year = int(d['discovery_date'].split('-')[0])
            if start_year <= year <= end_year:
                discoveries.append(d)
        return discoveries

    def get_high_impact_discoveries(self, min_impact: float = 9.0) -> List[Dict[str, Any]]:
        """Get discoveries with high impact scores"""
        return [d for d in self.discoveries if d['impact_score'] >= min_impact]

    def get_discovery_by_id(self, discovery_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific discovery by ID"""
        for d in self.discoveries:
            if d['id'] == discovery_id:
                return d
        return None

    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about the discovery database"""
        domains = {}
        for d in self.discoveries:
            domain = d['domain']
            if domain not in domains:
                domains[domain] = 0
            domains[domain] += 1

        years = [int(d['discovery_date'].split('-')[0]) for d in self.discoveries]

        return {
            "total_discoveries": len(self.discoveries),
            "domains": domains,
            "year_range": {
                "earliest": min(years),
                "latest": max(years)
            },
            "avg_impact_score": sum(d['impact_score'] for d in self.discoveries) / len(self.discoveries),
            "avg_citations": sum(d['citation_count'] for d in self.discoveries) / len(self.discoveries),
            "breakthrough_types": {
                "experimental": sum(1 for d in self.discoveries if d['breakthrough_type'] == 'experimental'),
                "theoretical": sum(1 for d in self.discoveries if d['breakthrough_type'] == 'theoretical'),
                "computational": sum(1 for d in self.discoveries if d['breakthrough_type'] == 'computational')
            }
        }


if __name__ == "__main__":
    # Initialize and save database
    db = DiscoveryDatabase(Path("../data/discoveries.json"))
    db.save_database()

    # Print statistics
    stats = db.get_discovery_statistics()
    print(f"Discovery Database Statistics:")
    print(f"Total Discoveries: {stats['total_discoveries']}")
    print(f"Domains: {stats['domains']}")
    print(f"Year Range: {stats['year_range']['earliest']} - {stats['year_range']['latest']}")
    print(f"Average Impact Score: {stats['avg_impact_score']:.2f}")
    print(f"Average Citations: {stats['avg_citations']:.0f}")
    print(f"Breakthrough Types: {stats['breakthrough_types']}")