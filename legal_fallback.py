"""
Fallback knowledge base for common legal questions
"""
from typing import Optional, Dict

COMMON_LEGAL_ANSWERS = {
    "minimum_wage": """
**What is the minimum wage for labor in Pakistan?**

**Current Minimum Wage (2025):**

**Federal Minimum Wage:**
- **PKR 32,000 per month** (as of July 2024)
- This applies to all federal government employees

**Provincial Minimum Wages:**
| Province | Minimum Wage (Monthly) |
|----------|------------------------|
| Punjab | PKR 32,000 |
| Sindh | PKR 25,000 |
| Khyber Pakhtunkhwa | PKR 28,000 |
| Balochistan | PKR 25,000 |

**Different Categories:**
- **Unskilled Labor:** PKR 25,000 - 28,000/month
- **Skilled Labor:** PKR 30,000 - 45,000/month
- **IT Sector:** PKR 40,000+ /month

**Legal Framework:**
- **Minimum Wage Act, 2015** - Governs minimum wage laws
- **Labor Code, 2018** - Employee rights and protections

**Employer Obligations:**
1. Must pay at least minimum wage
2. Overtime must be paid at double rate
3. Annual revision of wages by government

**Penalties for Non-Compliance:**
- Fine up to PKR 50,000
- Back payment of wages
- Legal action by labor courts

**Important Notes:**
- Wages are revised annually
- Different provinces may have different rates
- Some industries have higher wage standards
- You can file a complaint with the Labor Department

*This is for informational purposes only. Please consult the Labor Department for current rates.*
""",
    "polygamy": """
**Can a man marry a second wife in Pakistan?**

Yes, a man can marry a second wife in Pakistan, but with certain legal conditions:

**Legal Framework:**
- Under the Muslim Family Laws Ordinance, 1961, a man can marry a second wife
- However, he must obtain permission from the Arbitration Council
- The Arbitration Council must be satisfied that the marriage is necessary and just

**Conditions:**
1. The man must submit a written application to the Chairman of the Arbitration Council
2. The application must state the reasons for wanting a second marriage
3. The Arbitration Council must give permission

**Section Reference:**
- Muslim Family Laws Ordinance, 1961, Section 6: Polygamy

**Important:** This is a general overview. For specific legal advice, consult a qualified lawyer.
""",
    "domestic_violence": """
**Can a person beat his wife in Pakistan?**

**No, domestic violence is illegal in Pakistan.**

**Legal Protections:**
1. **Pakistan Penal Code** - Sections 493-498 deal with offenses relating to marriage
2. **Protection of Women (Criminal Laws Amendment) Act 2006** - Strengthened laws against domestic violence
3. **Domestic Violence (Prevention and Protection) Act 2012** - Provides protection orders and legal remedies

**Punishments:**
- Physical assault: Imprisonment up to 3 years and/or fine
- Causing hurt: Imprisonment up to 1 year and/or fine
- Grievous hurt: Imprisonment up to 7 years and/or fine

**Available Legal Remedies:**
- Protection order from the court
- Residence order
- Monetary compensation
- Custody of children

**Important:** If you or someone you know is experiencing domestic violence, contact:
- National Commission on the Status of Women
- Local police station
- Legal aid organizations

*This is for informational purposes only. Please consult a lawyer for legal advice.*
""",
    "murder": """
**What is the punishment for murder in Pakistan?**

**Primary Law:**
Under Section 302 of the Pakistan Penal Code (PPC), the punishment for murder (Qatl-i-amd) is:

**Punishments:**
1. **Death penalty** (qisas - death)
2. **Imprisonment for life** (if qisas is not applicable)
3. **Imprisonment up to 25 years** (in some cases)

**Categories of Murder:**
- **Qatl-i-amd (Intentional Murder)**: Section 302 - Death or life imprisonment
- **Qatl-i-shubh (Culpable Homicide)**: Section 315 - Up to 25 years imprisonment
- **Qatl-i-khata (Accidental Death)**: Section 316 - Up to 10 years imprisonment

**Key References:**
- Pakistan Penal Code, Section 302: Punishment for qatl-i-amd
- Pakistan Penal Code, Section 315: Qatl shibh-i-amd
- Pakistan Penal Code, Section 316: Qatl-i-khata

**Note:** The punishment can vary based on:
- Intent of the accused
- Circumstances of the crime
- Whether qisas (retaliation) is applicable
- Whether the heirs of the victim forgive the accused

*This is for informational purposes only. Please consult a lawyer for specific legal advice.*
""",
    "divorce": """
**What is Talaq (divorce) in Pakistan?**

**Definition:**
Talaq refers to the act of divorce in a Muslim marriage, where a man pronounces the words of divorce to his wife.

**Legal Framework:**
Under the Muslim Family Laws Ordinance, 1961:

**Procedure for Talaq:**
1. The man must pronounce the words of talaq
2. He must give written notice to the Chairman of the Arbitration Council
3. A copy of the notice must be sent to the wife
4. The talaq becomes effective after 90 days (iddat period)

**Section Reference:**
- Muslim Family Laws Ordinance, 1961, Section 7: Talaq

**Khula (Wife-initiated divorce):**
- A wife can seek divorce through khula
- She must apply to the court
- She may have to return her dower (mehr)

**Key Details:**
- The Arbitration Council tries to reconcile the parties
- If reconciliation fails, the divorce is finalized

*This is for informational purposes only. Please consult a lawyer for legal advice.*
""",
    "inheritance": """
**What are the laws regarding inheritance in Pakistan?**

**Legal Framework:**
Inheritance in Pakistan is governed by:
1. **Muslim Personal Law** (for Muslims)
2. **Succession Act, 1925** (for non-Muslims)
3. **Pakistan Penal Code, Section 330-338** (in some cases)

**For Muslims:**
- Inheritance follows the principles of Islamic law (Shariah)
- Fixed shares are given to specific heirs (spouse, children, parents)
- The remaining property is distributed among other heirs

**Key Principles:**
- A son receives twice the share of a daughter
- A wife receives 1/8 or 1/4 of the estate (depending on children)
- A husband receives 1/4 or 1/2 of the estate

**For Non-Muslims:**
- Governed by the Succession Act, 1925
- Property can be passed through a will (wasiyyat)

**Important:** Inheritance laws can be complex. Consult a lawyer for specific cases.

*This is for informational purposes only. Please consult a lawyer for legal advice.*
""",
    "marriage_age": """
**What is the minimum age for marriage in Pakistan?**

**Legal Age for Marriage:**

**For Muslims:**
- **Minimum age: 16 years** for both male and female (as per Muslim Family Laws Ordinance, 1961)
- However, the court can allow marriage below 16 years in exceptional circumstances

**For Non-Muslims:**
- **Child Marriage Restraint Act, 1929**: Minimum age is 18 for males and 16 for females

**Restrictions:**
- Section 2 of the Child Marriage Restraint Act, 1929 prohibits child marriage
- Punishment: Up to 1 month imprisonment and/or fine

**Important:**
- Marriage below the legal age requires court permission
- Consent of both parties is required

*This is for informational purposes only. Please consult a lawyer for legal advice.*
""",
    "theft": """
**What is the punishment for theft in Pakistan?**

**Primary Law:**
Under the Pakistan Penal Code (PPC):

**Section 379: Punishment for theft**
- Imprisonment of either description for a term up to 3 years
- Or fine
- Or both

**Aggravated Theft:**
**Section 381: Theft by clerk or servant**
- Imprisonment up to 7 years
- And shall also be liable to fine

**Section 381-A: Theft of motor vehicle**
- Imprisonment up to 7 years
- Fine not exceeding the value of the stolen vehicle

**Stolen Property:**
- Receiving stolen property: Section 411 - Up to 3 years imprisonment
- Habitual receiving: Section 413 - Up to 7 years imprisonment

*This is for informational purposes only. Please consult a lawyer for specific legal advice.*
"""
}

def get_fallback_answer(question: str) -> Optional[str]:
    """
    Check if the question matches a common legal query
    """
    question_lower = question.lower()
    
    # Check for polygamy/marriage questions
    if any(word in question_lower for word in ['second wife', '2nd wife', 'polygamy', 'multiple wives', 'more than one wife', 'second marriage']):
        return COMMON_LEGAL_ANSWERS['polygamy']
    
    # Check for domestic violence questions
    if any(word in question_lower for word in ['beat', 'hit', 'abuse', 'violence', 'hurt', 'injure']) and any(word in question_lower for word in ['wife', 'spouse', 'husband', 'domestic']):
        return COMMON_LEGAL_ANSWERS['domestic_violence']
    
    # Check for murder questions
    if any(word in question_lower for word in ['murder', 'kill', 'death', 'homicide', 'qatl']):
        return COMMON_LEGAL_ANSWERS['murder']
    
    # Check for divorce/talaq questions
    if any(word in question_lower for word in ['talaq', 'divorce', 'khula', 'separation']):
        return COMMON_LEGAL_ANSWERS['divorce']
    
    # Check for inheritance questions
    if any(word in question_lower for word in ['inheritance', 'heir', 'will', 'succession', 'property', 'estate']):
        return COMMON_LEGAL_ANSWERS['inheritance']
    
    # Check for marriage age questions
    if any(word in question_lower for word in ['marriage age', 'age of marriage', 'legal age', 'minimum age']):
        return COMMON_LEGAL_ANSWERS['marriage_age']
    
    # Check for theft questions
    if any(word in question_lower for word in ['theft', 'steal', 'robbery', 'stolen', 'dacoity']):
        return COMMON_LEGAL_ANSWERS['theft']
    
    return None