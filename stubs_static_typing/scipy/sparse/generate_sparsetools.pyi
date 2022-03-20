from typing import Any

BSR_ROUTINES: str
CSC_ROUTINES: str
CSR_ROUTINES: str
OTHER_ROUTINES: str
COMPILATION_UNITS: Any
I_TYPES: Any
T_TYPES: Any
THUNK_TEMPLATE: str
METHOD_TEMPLATE: str
GET_THUNK_CASE_TEMPLATE: str

def get_thunk_type_set(): ...
def parse_routine(name, args, types): ...
def main() -> None: ...
def write_autogen_blurb(stream) -> None: ...
