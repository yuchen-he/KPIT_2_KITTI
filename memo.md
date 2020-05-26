■KPIT format:

└── Dataset-1A
    └── Input-1
        ├── Scenario-1
        │   ├── D1A_input_1_S1.json
        │   ├── Input_Frames
        │       ├── 00001.png
        │       ├── 00002.png
        │       ├── ...
        └── Scenario-2
            ├──...
        └── ...
    └── Input-2
        ├── Scenario-1
        │   ├── D1A_input_2_S1.json
        │   ├── Input_Frames
        └── Scenario-2
            ├──...
        └── ...
└── Dataset-1B
    └── Input-1
    └── Input-2
    └── ...


■.json format:
├── Dataset-1A
│   └── Input-1
│       ├── calib
│       │   ├── 00001.txt
│       │   ├── 00002.txt
│       │   ├── ...
│       ├── labels
│       │   ├── D1A_input_1_S1_00001.txt
│       │   ├── D1A_input_1_S1_00002.txt
│       │   ├── ...
│       ├── resized_images
│       │   ├── 00001.png
│       │   ├── 00002.png
│       │   ├── ...
│       └── train.txt
