API_KEY = "qNfuofnOkcKJqMGeKvMfFFIJhngBcFXebanj0tr8"

JSON_PACKAGES = "packages"
JSON_ID = "packageId"
JSON_TITLE = "title"
JSON_DATES = "heldDates"
JSON_DATES2 = "dateIssued"

HEARING_TITLE = "title"
HEARING_ID = "id"
HEARING_DATES = "dates"
HEARING_TEXT = "text"
HEARING_CONGRESS = "congress"
HEARING_TURNS = "turns"
TURN_ID = "id"
TURN_SPEAKER = "speaker"
TURN_SPEAKER_FULL = "speaker_full"
TURN_SPEAKER_TYPE = "speaker_type"
TURN_IS_QUESTION = "is_question"
TURN_IS_ANSWER = "is_answer"

TURN_SPEAKER_TYPE_WITNESS = "witness"
TURN_SPEAKER_TYPE_POLITICIAN = "politician"

# from name parser: https://pypi.org/project/name_tools/
TITLES = set(
        [u'MSgt.', u'MSgt', u'Coach', u'Founder', u'Manager', u'Legal', u'Rebbe', u'Chair', u'Chairman',
         u'Chairwoman', u'CHAIRWOMAN', u'CHAIRMAN', u'Captain', u'Ballet',
         u'Baron', u'Father', u'Literary', u'Keyboardist', u'CCMSgt.', u'CCMSgt', u'Merchant', u'Adviser', u'Dutchess',
         u'Lamido', u'Mag. Judge', u'Surgeon', u'Missionary', u'Prefect', u'Magnate', u'Scholar', u'Investigator',
         u'Excellency', u'Celebrity', u'Brother', u'Delegate', u'Judicial', u'Dir.', u'CFO', u'Sultana', u'Docent',
         u'Chef', u'Honourable', u'Lawyer', u'7th', u'Subaltern', u'Business', u'2nd Lt', u'2nd Lt.', u'Hereditary',
         u'Nurse', u'Jurist', u'Admiral', u'9th', u'Clerk', u'Theorist', u'Ranger', u'Baseball', u'Nanny', u'Abbess',
         u'Dramatist', u'Teacher', u'Knowledge', u'Cyclist', u'Publisher', u'Comptroller', u'MCPOCG', u'Technical',
         u'Envoy', u'United', u'Credit', u'Musicologist', u'Advertising', u'Social', u'Dra.', u'Military', u'CMSgt',
         u'CMSgt.', u'Family', u'Deputy', u'Courtier', u'Sgt', u'Sgt.', u'Private', u'SGM', u'Composer', u'1st',
         u'Bandleader', u'Army', u'Archbishop', u'Archdruid', u'Sysselmann', u'Ayatollah', u'MSG', u'Pres.', u'Baba',
         u'PFC', u'LCDR', u'Biblical', u'CWO2', u'CW2', u'Musician', u'Heir', u'Flag', u'Excellent', u'Commander',
         u'Alderman', u'Chaplain', u'MD', u'M.D.', u'MG', u'Primate', u'Patriarch', u'Ms.', u'Mr.', u'Entertainer',
         u'Giani', u'Mufti', u'Suffragist', u'Division', u'Tax', u'High', u'Critic', u'CPO', u'SPC', u'Botanist',
         u'Risk', u'CSM', u'Sir', u'Lama', u'Guru', u'Hon.', u'Effendi', u'WO-1', u"King's", u'Drummer', u'Cardinal',
         u'LtG', u'Banker', u'Edohen', u'Designer', u'Customs', u'4th', u'MAG', u'President', u'Law', u'Sr.', u'Doctor',
         u'Psychologist', u'Presiding', u'Chief', u'SN', u'SA', u'Travel', u'SE', u'Producer', u'Rabbi', u'Tsarina',
         u'Gyani', u'Scientist', u'Comtesse', u'Mayor', u'Developer', u'Superior', u'Archdeacon', u'Verderer',
         u'Theologian', u'Dr.', u'Councillor', u'Maid', u'Lt', u'Lt.', u'Ens', u'Ens.', u'Co-Chairs', u'Criminal',
         u'FAdm', u'CEO', u'Goodwife', u'Comedienne', u'Brigadier', u'Commodore', u'BGen', u'Investor', u'Mystery',
         u'Mathematician', u'Naturalist', u'Curator', u'Shehu', u'Neuroscientist', u'Rock', u'Maharajah', u'Financial',
         u'Catholicos', u'Group', u'Navy', u'Blues', u'Adjutant', u'Collector', u'Eminence', u'Special', u'Rt',
         u'Shayk', u'1Sgt', u'3rd', u'Miss', u'Rep.', u'Rev,', u'VAdm', u'Reverend', u'Misses', u'Activist', u'Lord',
         u'Honorable', u'SMA', u'Associate', u'Marquise', u'Mme.', u'Princess', u'Barrister', u'Monsignor', u'British',
         u'Sheikh', u'Registrar', u'Generalissimo', u'Hajji', u'First', u'Tirthankar', u'Mademoiselle', u'Playwright',
         u'Revenue', u'Researcher', u'Blogger', u'LtJG', u'SMSgt', u'SMSgt.', u'Elder', u'Sailor', u'Comic',
         u'Paleontologist', u'Co-Founder', u'Engineer', u'Corporal', u'MaJ', u'District', u'5th', u'Historian',
         u'Master', u'Sergeant', u'Burgess', u'Saint', u'Edmi', u'Solicitor', u'Burlesque', u'Treasurer',
         u'Correspondent', u'MCPOC', u'MCPON', u'Inventor', u'King', u'Minister', u'Cartoonist', u'States',
         u'Architect', u'6th', u'Counselor', u'Countess', u'Printmaker', u'Anthropologist', u'Pro', u'Premier',
         u'Maharani', u'Comedian', u'Host', u'Tsar', u'SCPO', u'Goodman', u'Appellate', u'Educator', u'Pianist',
         u'CWO5', u'Lecturer', u'Evangelist', u'Printer', u'Matriarch', u'Theatre', u'Exec.', u'English', u'Pharaoh',
         u'MajGen', u'Most', u'Assoc.', u'Librarian', u'Mullah', u'Screenwriter', u'Presbyter', u'Singer', u'Duchesse',
         u'Docket', u'Professor', u'Mrs.', u'Deacon', u'Aunt', u'Colonel', u'Marchess', u'Businessman', u'Senior',
         u'LtC', u'Detective', u'Pope', u'Prin.', u'Queen', u'Sheik', u'Briggen', u'Television ', u'Radio',
         u'Industrialist', u'Economist', u'Principal', u'Archeologist', u'Sheriff', u'Writer', u'Philantropist',
         u'Historien', u'Sainte', u'Apprentice', u'Headman', u'Personality', u'DO', u'D.O.', u'Mister', u'His',
         u'Psychiatrist', u'Assistant', u'Designated', u'Ecologist', u'Mgr.', u'Singer-songwriter', u'Magistrate',
         u'SSg', u'SSg.', u'Banner', u'Gen', u'Gen.', u'Prime', u'Businesswoman', u'Vizier', u'CWO2', u'Srta.',
         u'Linguist', u'Graf', u'Secretary', u'1st Lt', u'1st Lt', u'Pvt', u'Pvt.', u'Choreographer', u'Intelligence',
         u'National', u'Memoirist', u'TSgt', u'Analytics', u'Computer', u'Bard', u'Marchioness', u'Marquess',
         u'Compositeur', u'Arhat', u'Expert', u'Federal', u'RAdm', u'Magistrate-Judge', u'Obstetritian', u'Discovery',
         u'Cartographer', u'PV2', u'Criminologist', u'Archduke', u'WM', u'Prior', u'Physicist', u'Jr', u'Jr.', u'Adept',
         u'Police', u'10th', u'Almoner', u'WO5', u'WO4', u'WO1', u'Priestess', u'WO3', u'Foreign', u'Award-winning',
         u'Col', u'Col.', u'Author', u'Majesty', u'Attache', u'LtCol', u'Seigneur', u'2nd', u'Dancer', u'GySgt',
         u'Biographer', u'Technologist', u'Shaykh', u'Petty', u'Shaikh', u'Strategy', u'Arbitrator', u'Poet', u'SSgt',
         u'SSgt.', u'Dame', u'Imam', u'Acolyte', u'PO3', u'PO1', u'Controller', u'Representative', u'Gaf',
         u'Instructor', u'Dpty', u'Dpty.', u'Painter', u'Pilot', u'Physician', u'Soccer', u'Politician', u'Consultant',
         u'Sultan', u"Charg\xe9 d'affaires", u'Governor', u'Air', u'CMSAF', u'Voice', u'Abbot', u'Elerunwon', u'VC',
         u'V.C.', u'Metropolitan', u'Resident', u'Attach\xe9', u'Canon', u'Dissident', u'Monk', u'Player', u'Tenor',
         u'WO2', u'Co-Chair', u'Soldier', u'Sociologist', u'Member', u'Mobster', u'Speaker', u'Grand', u'Essayist',
         u'Biochemist', u'Marcher', u'PhD', u'Ph.D.', u'Director', u'Warden', u'Senator', u'Vocalist', u'Priest',
         u'Theater', u'Mlle.', u'Bailiff', u'Academic', u'Mother', u'mModel', u'Corporate', u'Madame', u'Ambassador',
         u'Bearer', u'Madam', u'Executive', u'Actress', u'Biologist', u'Holiness', u'Prince', u'Pursuivant',
         u'Clergyman', u'Swordbearer', u'Photographer', u'LtGen', u'Lt. Gen.', u'Royal', u'Schoolmaster', u'Civil',
         u'Bench', u'SgtMaj', u'Chieftain', u'Doyen', u'Prelate', u'CDr', u'Adm', u'Adm.', u'Warrant', u'Kingdom',
         u'Lyricist', u'Municipal', u'Amn', u'Capt', u'Capt.', u'Chancellor', u'Advocate', u'Forester', u'Senior-Judge',
         u'Judge', u'Anarchist', u'Lady', u'Rear', u'lCpl', u'Chairs', u'Akhoond', u'Servant', u'Broadcaster',
         u'Journalist', u'Friar', u'Security', u'Attorney', u'Right', u'Classical', u'Staff', u'Astronomer', u'Shaik',
         u'Abolitionist', u'Mountaineer', u'Novelist', u'1stSgt', u'Philosopher', u'8th', u'Pioneer', u'Buddha',
         u'Prof.', u'Leader', u'Officer', u'MGySgt', u'BG', u'Archduchess', u'SgtMajMC', u'Marketing', u'Ornithologist',
         u'Lieutenant', u'Journeyman', u'Political', u'CWO-3', u'Translator', u'Sister', u'Sra.', u'CWO-5', u'CWO-4',
         u'Gentiluomo', u'Subedar', u'Pediatrician', u'Emperor', u'Software', u'Cheikh', u'Duke', u'Vicar', u'Auntie',
         u'Intendant', u'1Lt', u'Blessed', u'Empress', u'Entrepreneur', u'Saoshyant', u'Her', u'Zoologist', u'Flying',
         u'SFC', u'Bookseller', u'Editor', u'Narrator', u'Pastor', u'Soprano', u'Uncle', u'Junior', u'Highness',
         u'Count', u'Illustrator', u'Marquis', u'Siddha', u'CWO3', u'PSLC', u'Actor ', u'Vardapet', u'CWO4', u'Swami',
         u'Arranger', u'UK', u'Heiress', u'Asst', u'MCPO', u'Rangatira', u'Supreme', u'Ab', u'Opera', u'General',
         u'Provost', u"Queen's", u'Historicus', u'A1C', u'Pir', u'Bishop', u'Film', u'Commander-in-chief', u'Diplomat',
         u'Conductor', u'Operating', u'Bodhisattva', u'Guitarist', u'Bwana', u'Murshid', u'Field', u'Shekh',
         u'Mathematics', u'Wing', u'Chemist', u'Satirist', u'Woodman', u'Venerable', u'PO2', u'Druid', u'Mahdi',
         u'RDML', u'Viscount', u'Bibliographer', u'Cpl', u'Ekegbian', u'Vice', u'Behavioral', u'Timi', u'Cpt',
         u'Animator'])

