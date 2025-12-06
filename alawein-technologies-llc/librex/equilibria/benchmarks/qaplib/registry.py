"""
QAPLIB Instance Registry

Contains metadata for all 138 QAPLIB benchmark instances.
Each entry includes size, optimal value (if known), best known value,
and problem class information.

Reference: https://qaplib.mgi.polymtl.ca/
"""

from typing import Dict, List, Optional, Union


class QAPLIBInstance:
    """Metadata for a single QAPLIB instance"""

    def __init__(
        self,
        name: str,
        size: int,
        optimal_value: Optional[int] = None,
        best_known_value: Optional[int] = None,
        problem_class: str = "unknown",
        description: str = "",
        url: Optional[str] = None,
    ):
        self.name = name
        self.size = size
        self.optimal_value = optimal_value
        self.best_known_value = best_known_value or optimal_value
        self.problem_class = problem_class
        self.description = description
        self.url = url or f"https://qaplib.mgi.polymtl.ca/data/qapdata/{name}.dat"

    def is_optimal_known(self) -> bool:
        """Check if optimal solution is known"""
        return self.optimal_value is not None

    def get_value(self) -> Optional[int]:
        """Get best available value (optimal or best known)"""
        return self.optimal_value or self.best_known_value

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "size": self.size,
            "optimal_value": self.optimal_value,
            "best_known_value": self.best_known_value,
            "problem_class": self.problem_class,
            "description": self.description,
            "url": self.url
        }


# Complete registry of all 138 QAPLIB instances
QAPLIB_REGISTRY: Dict[str, QAPLIBInstance] = {
    # Christofides-Brusco instances (size 12-30)
    "chr12a": QAPLIBInstance("chr12a", 12, 9552, problem_class="grid",
                            description="Christofides keyboard layout"),
    "chr12b": QAPLIBInstance("chr12b", 12, 9742, problem_class="grid"),
    "chr12c": QAPLIBInstance("chr12c", 12, 11156, problem_class="grid"),
    "chr15a": QAPLIBInstance("chr15a", 15, 9896, problem_class="grid"),
    "chr15b": QAPLIBInstance("chr15b", 15, 7990, problem_class="grid"),
    "chr15c": QAPLIBInstance("chr15c", 15, 9504, problem_class="grid"),
    "chr18a": QAPLIBInstance("chr18a", 18, 11098, problem_class="grid"),
    "chr18b": QAPLIBInstance("chr18b", 18, 1534, problem_class="grid"),
    "chr20a": QAPLIBInstance("chr20a", 20, 2192, problem_class="grid"),
    "chr20b": QAPLIBInstance("chr20b", 20, 2298, problem_class="grid"),
    "chr20c": QAPLIBInstance("chr20c", 20, 14142, problem_class="grid"),
    "chr22a": QAPLIBInstance("chr22a", 22, 6156, problem_class="grid"),
    "chr22b": QAPLIBInstance("chr22b", 22, 6194, problem_class="grid"),
    "chr25a": QAPLIBInstance("chr25a", 25, 3796, problem_class="grid"),

    # Els instances (size 19)
    "els19": QAPLIBInstance("els19", 19, 17212548, problem_class="real-world",
                           description="Electronics manufacturing"),

    # Eschermann-Schweitzer instances (size 16-64)
    "esc16a": QAPLIBInstance("esc16a", 16, 68, problem_class="structured"),
    "esc16b": QAPLIBInstance("esc16b", 16, 292, problem_class="structured"),
    "esc16c": QAPLIBInstance("esc16c", 16, 160, problem_class="structured"),
    "esc16d": QAPLIBInstance("esc16d", 16, 16, problem_class="structured"),
    "esc16e": QAPLIBInstance("esc16e", 16, 28, problem_class="structured"),
    "esc16f": QAPLIBInstance("esc16f", 16, 0, problem_class="structured"),
    "esc16g": QAPLIBInstance("esc16g", 16, 26, problem_class="structured"),
    "esc16h": QAPLIBInstance("esc16h", 16, 996, problem_class="structured"),
    "esc16i": QAPLIBInstance("esc16i", 16, 14, problem_class="structured"),
    "esc16j": QAPLIBInstance("esc16j", 16, 8, problem_class="structured"),
    "esc32a": QAPLIBInstance("esc32a", 32, 130, problem_class="structured"),
    "esc32b": QAPLIBInstance("esc32b", 32, 168, problem_class="structured"),
    "esc32c": QAPLIBInstance("esc32c", 32, 642, problem_class="structured"),
    "esc32d": QAPLIBInstance("esc32d", 32, 200, problem_class="structured"),
    "esc32e": QAPLIBInstance("esc32e", 32, 2, problem_class="structured"),
    "esc32f": QAPLIBInstance("esc32f", 32, 2, problem_class="structured"),
    "esc32g": QAPLIBInstance("esc32g", 32, 6, problem_class="structured"),
    "esc32h": QAPLIBInstance("esc32h", 32, 438, problem_class="structured"),
    "esc64a": QAPLIBInstance("esc64a", 64, 116, problem_class="structured"),
    "esc128": QAPLIBInstance("esc128", 128, 64, problem_class="structured"),

    # Had instances (size 4-20)
    "had4": QAPLIBInstance("had4", 4, 8, problem_class="Hadamard"),
    "had6": QAPLIBInstance("had6", 6, 54, problem_class="Hadamard"),
    "had8": QAPLIBInstance("had8", 8, 136, problem_class="Hadamard"),
    "had10": QAPLIBInstance("had10", 10, 310, problem_class="Hadamard"),
    "had12": QAPLIBInstance("had12", 12, 1652, problem_class="Hadamard"),
    "had14": QAPLIBInstance("had14", 14, 2724, problem_class="Hadamard"),
    "had16": QAPLIBInstance("had16", 16, 3720, problem_class="Hadamard"),
    "had18": QAPLIBInstance("had18", 18, 5358, problem_class="Hadamard"),
    "had20": QAPLIBInstance("had20", 20, 6922, problem_class="Hadamard"),

    # Krarup instances (size 30-32)
    "kra30a": QAPLIBInstance("kra30a", 30, 88900, problem_class="real-world",
                            description="Danish hospital layout"),
    "kra30b": QAPLIBInstance("kra30b", 30, 91420, problem_class="real-world"),
    "kra32": QAPLIBInstance("kra32", 32, 88700, problem_class="real-world"),

    # Lipa instances (size 20-90)
    "lipa20a": QAPLIBInstance("lipa20a", 20, 3683, problem_class="generated"),
    "lipa20b": QAPLIBInstance("lipa20b", 20, 27076, problem_class="generated"),
    "lipa30a": QAPLIBInstance("lipa30a", 30, 13178, problem_class="generated"),
    "lipa30b": QAPLIBInstance("lipa30b", 30, 151426, problem_class="generated"),
    "lipa40a": QAPLIBInstance("lipa40a", 40, 31538, problem_class="generated"),
    "lipa40b": QAPLIBInstance("lipa40b", 40, 476581, problem_class="generated"),
    "lipa50a": QAPLIBInstance("lipa50a", 50, 62093, problem_class="generated"),
    "lipa50b": QAPLIBInstance("lipa50b", 50, 1210244, problem_class="generated"),
    "lipa60a": QAPLIBInstance("lipa60a", 60, 107218, problem_class="generated"),
    "lipa60b": QAPLIBInstance("lipa60b", 60, 2520135, problem_class="generated"),
    "lipa70a": QAPLIBInstance("lipa70a", 70, 169755, problem_class="generated"),
    "lipa70b": QAPLIBInstance("lipa70b", 70, 4603200, problem_class="generated"),
    "lipa80a": QAPLIBInstance("lipa80a", 80, 253195, problem_class="generated"),
    "lipa80b": QAPLIBInstance("lipa80b", 80, 7763962, problem_class="generated"),
    "lipa90a": QAPLIBInstance("lipa90a", 90, 360630, problem_class="generated"),
    "lipa90b": QAPLIBInstance("lipa90b", 90, 12490441, problem_class="generated"),

    # Nug instances - Nugent et al. (size 12-30)
    "nug12": QAPLIBInstance("nug12", 12, 578, problem_class="real-world"),
    "nug14": QAPLIBInstance("nug14", 14, 1014, problem_class="real-world"),
    "nug15": QAPLIBInstance("nug15", 15, 1150, problem_class="real-world"),
    "nug16a": QAPLIBInstance("nug16a", 16, 1610, problem_class="real-world"),
    "nug16b": QAPLIBInstance("nug16b", 16, 1240, problem_class="real-world"),
    "nug17": QAPLIBInstance("nug17", 17, 1732, problem_class="real-world"),
    "nug18": QAPLIBInstance("nug18", 18, 1930, problem_class="real-world"),
    "nug20": QAPLIBInstance("nug20", 20, 2570, problem_class="real-world"),
    "nug21": QAPLIBInstance("nug21", 21, 2438, problem_class="real-world"),
    "nug22": QAPLIBInstance("nug22", 22, 3596, problem_class="real-world"),
    "nug24": QAPLIBInstance("nug24", 24, 3488, problem_class="real-world"),
    "nug25": QAPLIBInstance("nug25", 25, 3744, problem_class="real-world"),
    "nug27": QAPLIBInstance("nug27", 27, 5234, problem_class="real-world"),
    "nug28": QAPLIBInstance("nug28", 28, 5166, problem_class="real-world"),
    "nug30": QAPLIBInstance("nug30", 30, 6124, problem_class="real-world"),

    # Roucairol instances (size 12-20)
    "rou12": QAPLIBInstance("rou12", 12, 235528, problem_class="real-world"),
    "rou15": QAPLIBInstance("rou15", 15, 354210, problem_class="real-world"),
    "rou20": QAPLIBInstance("rou20", 20, 725522, problem_class="real-world"),

    # Scriabin instances (size 12-20)
    "scr12": QAPLIBInstance("scr12", 12, 31410, problem_class="real-world"),
    "scr15": QAPLIBInstance("scr15", 15, 51140, problem_class="real-world"),
    "scr20": QAPLIBInstance("scr20", 20, 110030, problem_class="real-world"),

    # Skorin-Kapov instances (size 42-100)
    "sko42": QAPLIBInstance("sko42", 42, 15812, problem_class="random"),
    "sko49": QAPLIBInstance("sko49", 49, 23386, problem_class="random"),
    "sko56": QAPLIBInstance("sko56", 56, 34458, problem_class="random"),
    "sko64": QAPLIBInstance("sko64", 64, 48498, problem_class="random"),
    "sko72": QAPLIBInstance("sko72", 72, 66256, problem_class="random"),
    "sko81": QAPLIBInstance("sko81", 81, 90998, problem_class="random"),
    "sko90": QAPLIBInstance("sko90", 90, 115534, problem_class="random"),
    "sko100a": QAPLIBInstance("sko100a", 100, 152002, problem_class="random"),
    "sko100b": QAPLIBInstance("sko100b", 100, 153890, problem_class="random"),
    "sko100c": QAPLIBInstance("sko100c", 100, 147862, problem_class="random"),
    "sko100d": QAPLIBInstance("sko100d", 100, 149576, problem_class="random"),
    "sko100e": QAPLIBInstance("sko100e", 100, 149150, problem_class="random"),
    "sko100f": QAPLIBInstance("sko100f", 100, 149036, problem_class="random"),

    # Steinberg instances (size 36)
    "ste36a": QAPLIBInstance("ste36a", 36, 9526, problem_class="grid",
                            description="Steinberg backboard wiring"),
    "ste36b": QAPLIBInstance("ste36b", 36, 15852, problem_class="grid"),
    "ste36c": QAPLIBInstance("ste36c", 36, 8239110, problem_class="grid"),

    # Taillard instances (size 20-256)
    "tai12a": QAPLIBInstance("tai12a", 12, 224416, problem_class="random-structured"),
    "tai12b": QAPLIBInstance("tai12b", 12, 39464925, problem_class="random-structured"),
    "tai15a": QAPLIBInstance("tai15a", 15, 388214, problem_class="random-structured"),
    "tai15b": QAPLIBInstance("tai15b", 15, 51765268, problem_class="random-structured"),
    "tai17a": QAPLIBInstance("tai17a", 17, 491812, problem_class="random-structured"),
    "tai20a": QAPLIBInstance("tai20a", 20, 703482, problem_class="random-structured"),
    "tai20b": QAPLIBInstance("tai20b", 20, 122455319, problem_class="random-structured"),
    "tai25a": QAPLIBInstance("tai25a", 25, 1167256, problem_class="random-structured"),
    "tai25b": QAPLIBInstance("tai25b", 25, 344355646, problem_class="random-structured"),
    "tai30a": QAPLIBInstance("tai30a", 30, 1818146, problem_class="random-structured"),
    "tai30b": QAPLIBInstance("tai30b", 30, 637117113, problem_class="random-structured"),
    "tai35a": QAPLIBInstance("tai35a", 35, 2422002, problem_class="random-structured"),
    "tai35b": QAPLIBInstance("tai35b", 35, 283315445, problem_class="random-structured"),
    "tai40a": QAPLIBInstance("tai40a", 40, 3139370, problem_class="random-structured"),
    "tai40b": QAPLIBInstance("tai40b", 40, 637250948, problem_class="random-structured"),
    "tai50a": QAPLIBInstance("tai50a", 50, 4938796, problem_class="random-structured"),
    "tai50b": QAPLIBInstance("tai50b", 50, 458821517, problem_class="random-structured"),
    "tai60a": QAPLIBInstance("tai60a", 60, 7205962, problem_class="random-structured"),
    "tai60b": QAPLIBInstance("tai60b", 60, 608215054, problem_class="random-structured"),
    "tai64c": QAPLIBInstance("tai64c", 64, 1855928, problem_class="grid"),
    "tai80a": QAPLIBInstance("tai80a", 80, 13499184, problem_class="random-structured"),
    "tai80b": QAPLIBInstance("tai80b", 80, 818415043, problem_class="random-structured"),
    "tai100a": QAPLIBInstance("tai100a", 100, 21052466, problem_class="random-structured"),
    "tai100b": QAPLIBInstance("tai100b", 100, 1185996137, problem_class="random-structured"),
    "tai150b": QAPLIBInstance("tai150b", 150, best_known_value=498896643, problem_class="random-structured"),
    "tai256c": QAPLIBInstance("tai256c", 256, 44759294, problem_class="grid"),

    # Thonemann-Bolte instances (size 30-40)
    "tho30": QAPLIBInstance("tho30", 30, 149936, problem_class="random"),
    "tho40": QAPLIBInstance("tho40", 40, 240516, problem_class="random"),
    "tho150": QAPLIBInstance("tho150", 150, best_known_value=8133398, problem_class="random"),

    # Wilhelm-Ward instances (size 50)
    "wil50": QAPLIBInstance("wil50", 50, 48816, problem_class="random"),

    # Burkard-Offermann instances
    "bur26a": QAPLIBInstance("bur26a", 26, 5426670, problem_class="real-world"),
    "bur26b": QAPLIBInstance("bur26b", 26, 3817852, problem_class="real-world"),
    "bur26c": QAPLIBInstance("bur26c", 26, 5426795, problem_class="real-world"),
    "bur26d": QAPLIBInstance("bur26d", 26, 3821225, problem_class="real-world"),
    "bur26e": QAPLIBInstance("bur26e", 26, 5386879, problem_class="real-world"),
    "bur26f": QAPLIBInstance("bur26f", 26, 3782044, problem_class="real-world"),
    "bur26g": QAPLIBInstance("bur26g", 26, 10117172, problem_class="real-world"),
    "bur26h": QAPLIBInstance("bur26h", 26, 7098658, problem_class="real-world"),
}


def get_instance_by_size(min_size: int = 0, max_size: int = 300) -> List[str]:
    """Get all instances within a size range"""
    return [
        name for name, inst in QAPLIB_REGISTRY.items()
        if min_size <= inst.size <= max_size
    ]


def get_instance_by_class(problem_class: str) -> List[str]:
    """Get all instances of a specific problem class"""
    return [
        name for name, inst in QAPLIB_REGISTRY.items()
        if inst.problem_class == problem_class
    ]


def get_small_instances(max_size: int = 20) -> List[str]:
    """Get small instances suitable for testing"""
    return get_instance_by_size(0, max_size)


def get_instance_metadata(name: str) -> Optional[Dict]:
    """Get metadata for a specific instance"""
    if name in QAPLIB_REGISTRY:
        return QAPLIB_REGISTRY[name].to_dict()
    return None


def get_all_instance_names() -> List[str]:
    """Get all instance names"""
    return list(QAPLIB_REGISTRY.keys())


def get_problem_classes() -> List[str]:
    """Get all unique problem classes"""
    return list(set(inst.problem_class for inst in QAPLIB_REGISTRY.values()))