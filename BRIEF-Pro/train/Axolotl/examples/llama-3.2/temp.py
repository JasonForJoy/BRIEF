ValueError: Got a string but expected a list instead: 'John J. Collins is the Holmes Professor of Old Testament Criticism & Interpretation at Yale Divinity School.
Yale Divinity School was established in 1822.'
[2024-11-16 19:29:18,363] [DEBUG] [axolotl.callbacks.compute:449] [PID:346004] [RANK:0] Failed to compute metric sacrebleu with kwargs dict_keys(['references', 'predictions', 'sources'])
dict_items(
    [
        ('references', [
            ['John J. Collins is the Holmes Professor of Old Testament Criticism & Interpretation at Yale Divinity School.\nYale Divinity School was established in 1822.'], 
            ['Emperor Haile Selassie laid claim to both territories in a letter to Franklin D. Roosevelt at the Paris Peace Conference and at the First Session of the United Nations.\nTito exchanged visits (1954 and 1956) with Emperor Haile Selassie of Ethiopia.'], 
            ['Isabelle Huppert starred in Captive.\nThe Piano Teacher won the prestigious Grand Prize at the 2001 Cannes Film Festival.'], 
            ['CFGX-FM broadcasts at 99.9 FM in Sarnia, Ontario.\nPoint Edward is an independent village in Ontario, Canada.'], 
            ['On 17 December 1950, Kim Il-sung was deprived of the right of command of KPA by China.\nGeneral H. M. Zakharov advised Kim Il-sung to redeploy his forces to defend Seoul.'], 
            ["Derek Shepherd was driving to the airport for his final trip to Washington.\nThe eleventh season of the American television medical drama Grey's Anatomy premiered on September 25, 2014 in the United States."], 
            ["Sophie Ellis-Bextor is an English singer.\nSophie Ellis-Bextor's parents were Janet Ellis and Robin Bextor."], 
            ['Heart and Soul is the sixteenth studio album by Kenny G.\nThe University of Washington is located in Seattle.']]
            ), 
        ('predictions', 
         [
             '1. John J. Collins is the Holmes Professor of Old Testament Criticism & Interpretation at Yale Divinity School.\nThe Holmes Professorship was established in 1976.\nThe Holmes Professorship was established by the Yale Corporation.',
             "Tito's foreign policy led to relationships with a variety of governments.\nThe first 100 days of a first-term presidency of a President of the United States are sometimes used to measure the successes and accomplishments of a president during the time that the president's power and influence are at their greatest.",
             'Captive is a 2012 French-Filipino drama film directed by Brillante Mendoza.\nThe film starred Isabelle Huppert.', 
             "1. CFGX-FM is licensed to Sarnia, Ontario.\nThe station's studios are located in Sarnia.", 
             "The Chinese People's Volunteer Army was formed by the Red Army of the Communist Party of China.\nThe Chinese People's Volunteer Army was commanded by Peng Dehuai.", 
             "1. Meredith Grey is the series' protagonist.\n2. Meredith Grey is the daughter of world-renowned surgeon Ellis Grey.\n3. Meredith Grey was introduced as a surgical intern at the fictional Seattle Grace Hospital.",
            'The film stars Vincent Cassel as Paul.\nThe song was performed by the character Magenta.', 
            '1. Heart and Soul is the sixteenth studio album by Kenny G.\nThe album was released on June 29, 2010.\nThe song was written and produced solely by Babyface.'
        ]
        )
    ]
 )
[2024-11-16 19:29:18,370] [DEBUG] [axolotl.callbacks.compute:449] [PID:346004] [RANK:0] Failed to compute metric sacrebleu with kwargs dict_keys(['references', 'predictions'])
{'eval_chrf': {'score': 34.52022483798369, 'char_order': 6, 'word_order': 0, 'beta': 2}, 'eval_sacrebleu': {'score': 17.846068899420448, 'counts': [85, 42, 32, 27], 'totals': [247, 239, 231, 223], 'precisions': [34.412955465587046, 17.573221757322177, 13.852813852813853, 12.10762331838565], 'bp': 1.0, 'sys_len': 247, 'ref_len': 225}, 'epoch': 0.06}