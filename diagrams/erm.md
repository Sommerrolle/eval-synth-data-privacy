erDiagram 

insurance_data {
	int pid FK
}


insurant ||--o{ insurance_data : "1 .. n"

inpatient_diagnosis {
	int pid FK
	int case_id FK
}

inpatient_diagnosis }o--|| inpatient_cases : "n .. 1"

inpatient_fees {
    int pid FK
    int case_id FK
}
inpatient_fees }o--|| inpatient_cases : "n .. 1"

inpatient_procedures {
    int pid FK
    int case_id FK
}
inpatient_procedures }o--|| inpatient_cases : "n .. 1"

inpatient_cases {
	int pid FK
	int case_id PK
}
inpatient_cases }o--|| insurant : "n .. 1"

insurant {
	int pid PK
}

outpatient_cases {
	int pid FK
	int case_id PK
}
insurant ||--o{ outpatient_cases : "1 .. n"


outpatient_diagnosis {
	int pid FK
	int case_id FK
}
outpatient_cases ||--o{ outpatient_diagnosis : "1 .. n"


outpatient_fees {
	int pid FK
	int case_id FK
}
outpatient_cases ||--o{ outpatient_fees : "1 .. n"


outpatient_procedures {
	int pid FK
	int case_id FK
}
outpatient_cases ||--o{ outpatient_procedures : "1 .. n"

insurant ||--o{ drugs  : "1 .. n"
drugs {
	int pid FK
}
