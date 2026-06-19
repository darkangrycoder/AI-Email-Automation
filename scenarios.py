"""
scenarios.py
────────────
Ten evaluation scenarios spanning a deliberate cross-section of:
  • professional contexts (job search, client relations, HR, sales, support)
  • tones (formal, empathetic, urgent, persuasive, warm, neutral, firm)
  • fact-density levels (3–5 key facts each)

Each reference_email is a hand-crafted gold-standard output used by the
Email Quality Score (EQS) ROUGE-L sub-component.
"""

SCENARIOS = [
    {
        "id": 1,
        "intent": "Follow up on a submitted job application",
        "key_facts": [
            "Applied 2 weeks ago for Senior ML Engineer position",
            "Application submitted via the company careers portal",
            "Strong interest in the Natural Language Processing research team",
            "Available for interviews any time in the next 3 weeks",
        ],
        "tone": "Professional, enthusiastic",
        "reference_email": (
            "Subject: Following Up: Senior ML Engineer Application – [Your Name]\n\n"
            "Dear Hiring Team,\n\n"
            "I hope this message finds you well. I am writing to follow up on my "
            "application for the Senior ML Engineer position, submitted approximately "
            "two weeks ago through your careers portal.\n\n"
            "I remain genuinely enthusiastic about this opportunity, particularly given "
            "the innovative work your Natural Language Processing research team is "
            "pursuing. My background aligns closely with the challenges your team is "
            "tackling, and I am confident I could contribute meaningfully from day one.\n\n"
            "I am available for an interview at any point over the next three weeks and "
            "would welcome the chance to speak further. Please do not hesitate to reach "
            "out if additional information would be helpful.\n\n"
            "Thank you sincerely for your time and consideration.\n\n"
            "Best regards,\n[Your Name]"
        ),
    },
    {
        "id": 2,
        "intent": "Schedule a quarterly business review meeting with a key client",
        "key_facts": [
            "Client: Meridian Financial Group",
            "Q3 project deliverables are ready for review",
            "Proposed meeting dates: September 18, 19, or 20",
            "Meeting can be conducted virtually or in person at client's preference",
        ],
        "tone": "Formal, courteous",
        "reference_email": (
            "Subject: Q3 Business Review Meeting Request – Meridian Financial Group\n\n"
            "Dear Meridian Financial Group Team,\n\n"
            "I hope you are well. I am writing to propose scheduling our Quarterly "
            "Business Review to discuss the Q3 project deliverables, which are now "
            "ready for your review.\n\n"
            "I would like to suggest September 18, 19, or 20 as potential dates and am "
            "fully flexible regarding format — we are equally prepared to meet in person "
            "at your offices or to conduct the session virtually, whichever is more "
            "convenient for your team.\n\n"
            "Could you please advise on your preferred date and format at your earliest "
            "convenience? I will ensure all supporting materials are distributed to your "
            "team well in advance of the meeting.\n\n"
            "Thank you for your continued partnership. I look forward to a productive review.\n\n"
            "Yours sincerely,\n[Your Name]\n[Title] | [Company]"
        ),
    },
    {
        "id": 3,
        "intent": "Request a one-week project deadline extension from a manager",
        "key_facts": [
            "Original deadline: Friday, October 10",
            "Requesting extension to Friday, October 17",
            "Delay caused by an unexpected data pipeline failure on the ETL layer",
            "75% of the deliverable is already complete",
            "Will provide daily progress updates if extension is granted",
        ],
        "tone": "Urgent, apologetic, professional",
        "reference_email": (
            "Subject: Urgent: Extension Request – Project Deadline October 10\n\n"
            "Hi [Manager's Name],\n\n"
            "I want to flag an issue immediately. Due to an unexpected failure in our "
            "ETL data pipeline, we have encountered a blocker that was not foreseeable "
            "at the time of planning. I sincerely apologize for the impact this may "
            "have on the wider schedule.\n\n"
            "The deliverable is currently 75% complete. However, to ensure the remaining "
            "work meets the expected standard of quality, I am requesting a one-week "
            "extension, moving the deadline from October 10 to October 17.\n\n"
            "I understand the seriousness of this request. To maintain full transparency, "
            "I am prepared to send daily progress updates for the duration of the "
            "extension period.\n\n"
            "Please let me know if you would like to discuss this further. I remain fully "
            "committed to delivering a high-quality outcome.\n\n"
            "Best regards,\n[Your Name]"
        ),
    },
    {
        "id": 4,
        "intent": "Respond to a customer complaint about a delayed shipment",
        "key_facts": [
            "Order #47821 was placed on August 5",
            "Original promised delivery date: August 12",
            "Actual delivery date: August 19 (7 days late)",
            "Cause: Regional courier service disruption",
            "Offering 15% discount on the customer's next order as compensation",
        ],
        "tone": "Empathetic, apologetic",
        "reference_email": (
            "Subject: Sincere Apology – Order #47821 Delivery Delay\n\n"
            "Dear [Customer Name],\n\n"
            "Thank you for reaching out, and please accept my sincere apologies for the "
            "significant delay you experienced with Order #47821.\n\n"
            "Your order, placed on August 5 with an expected delivery of August 12, was "
            "regrettably delayed until August 19 due to a disruption affecting our "
            "regional courier service — a situation entirely outside the norm for our "
            "operations. I fully understand how frustrating this must have been, and I "
            "am truly sorry for the inconvenience caused.\n\n"
            "To make amends, I am pleased to offer you a 15% discount on your next "
            "order, automatically applied at checkout. Please consider this a small "
            "acknowledgment of the delay and our appreciation for your patience.\n\n"
            "Your experience matters to us, and we are working with our logistics "
            "partners to prevent a recurrence. Should you have any further questions, "
            "please do not hesitate to contact us.\n\n"
            "Warm regards,\n[Customer Care Team]\n[Company Name]"
        ),
    },
    {
        "id": 5,
        "intent": "Introduce a new executive hire to the entire company",
        "key_facts": [
            "New hire: Dr. Priya Sharma",
            "Role: Head of AI Research",
            "Start date: November 1",
            "Background: 10 years at DeepMind, PhD in Computer Science from MIT",
            "Will lead the company's new Foundation Models initiative",
        ],
        "tone": "Warm, enthusiastic, professional",
        "reference_email": (
            "Subject: Welcome Dr. Priya Sharma – Our New Head of AI Research\n\n"
            "Dear Team,\n\n"
            "I am thrilled to announce that Dr. Priya Sharma will be joining us on "
            "November 1 as our new Head of AI Research.\n\n"
            "Priya brings an exceptional depth of expertise — most recently spending "
            "10 years at DeepMind, and holding a PhD in Computer Science from MIT. "
            "Her work on large-scale AI systems is widely respected across the research "
            "community, and we are genuinely fortunate to have her on board.\n\n"
            "In her new role, Priya will be leading our Foundation Models initiative — "
            "an exciting strategic priority that I know many of you have been eager to "
            "see take shape. Her vision and experience will be instrumental in defining "
            "how we build and deploy AI at scale.\n\n"
            "Please join me in giving Priya a warm welcome when she joins us next month. "
            "I have no doubt she will make an extraordinary impact.\n\n"
            "Warm regards,\n[CEO / Department Head Name]"
        ),
    },
    {
        "id": 6,
        "intent": "Request a formal proposal from a software vendor for a CRM solution",
        "key_facts": [
            "Evaluating enterprise CRM solutions for the sales department",
            "Annual budget: up to $50,000",
            "Required integrations: Salesforce and Slack",
            "Target implementation date: Q1 next year",
            "Proposal submission deadline: October 31",
        ],
        "tone": "Formal, business-like",
        "reference_email": (
            "Subject: Request for Proposal – Enterprise CRM Solution\n\n"
            "Dear [Vendor Name] Sales Team,\n\n"
            "I am writing on behalf of [Company Name] to formally request a proposal "
            "for an enterprise CRM solution for our sales department.\n\n"
            "We are evaluating vendors with an annual budget of up to $50,000 and "
            "require seamless integration with our existing Salesforce and Slack "
            "environments. Our target implementation timeline is Q1 of next year, and "
            "we intend to complete vendor selection prior to that milestone.\n\n"
            "We would kindly request that your proposal address the following: a "
            "detailed feature overview, total cost of ownership, implementation "
            "timeline, and technical documentation for the required integrations.\n\n"
            "Please submit your proposal no later than October 31. We welcome a "
            "follow-up call to clarify requirements if needed.\n\n"
            "We look forward to reviewing your submission.\n\n"
            "Yours sincerely,\n[Your Name]\n[Title] | [Company]"
        ),
    },
    {
        "id": 7,
        "intent": "Send a thank-you note after a job interview",
        "key_facts": [
            "Interviewed for the Product Manager role at Nexus AI",
            "Interviewed with Tom Bradley (Hiring Manager) and Aisha Okonkwo (Team Lead)",
            "Discussed the roadmap for AI-powered analytics features",
            "Particularly excited about the company's Series B expansion plans",
        ],
        "tone": "Warm, professional, grateful",
        "reference_email": (
            "Subject: Thank You – Product Manager Interview at Nexus AI\n\n"
            "Dear Tom and Aisha,\n\n"
            "Thank you so much for taking the time to speak with me yesterday about the "
            "Product Manager role at Nexus AI. It was a genuinely energising conversation, "
            "and I left with an even stronger sense of excitement about the team and the "
            "direction the company is heading.\n\n"
            "I particularly enjoyed our discussion around the roadmap for AI-powered "
            "analytics features — the depth of ambition there resonates closely with the "
            "work I find most meaningful. Learning more about the Series B expansion plans "
            "only reinforced my enthusiasm for what Nexus AI is building.\n\n"
            "I believe my experience and approach would be a strong fit for this role, "
            "and I would be delighted to join the team. Please do not hesitate to reach "
            "out if any further information would be helpful in your decision-making process.\n\n"
            "Thank you again for a wonderful interview experience.\n\n"
            "Warm regards,\n[Your Name]"
        ),
    },
    {
        "id": 8,
        "intent": "Send a payment reminder for an overdue invoice",
        "key_facts": [
            "Invoice #INV-2024-0892 for $12,500",
            "Original payment due date: September 1",
            "Invoice is now 21 days overdue",
            "Services rendered: August website development project",
            "Payment can be made via bank transfer or through the client portal",
        ],
        "tone": "Firm but polite, professional",
        "reference_email": (
            "Subject: Payment Reminder – Invoice #INV-2024-0892 (21 Days Overdue)\n\n"
            "Dear [Client Name],\n\n"
            "I hope you are doing well. I am writing to follow up on Invoice "
            "#INV-2024-0892 for $12,500, issued for the August website development "
            "project. As of today, this invoice is 21 days past its original due date "
            "of September 1.\n\n"
            "I understand that oversights can happen and am confident this is simply one. "
            "However, I would be grateful if you could arrange payment at your earliest "
            "convenience. For your ease, payment can be made via bank transfer or "
            "directly through our client portal.\n\n"
            "If there are any discrepancies with the invoice or questions regarding the "
            "services rendered, please do not hesitate to reach out so we can resolve "
            "them promptly.\n\n"
            "Thank you for your attention to this matter. I look forward to hearing from you.\n\n"
            "Best regards,\n[Your Name]\n[Title] | [Company]"
        ),
    },
    {
        "id": 9,
        "intent": "Propose a strategic content syndication partnership",
        "key_facts": [
            "Target company: EduLearn Platform (2M+ monthly active users)",
            "Proposal: Content syndication partnership",
            "Key benefit: Expanding reach into the education market",
            "Structure: 3-month pilot period to begin",
            "Revenue sharing model: 30/70 split in EduLearn's favour",
        ],
        "tone": "Persuasive, confident, professional",
        "reference_email": (
            "Subject: Partnership Proposal: Content Syndication Opportunity for EduLearn\n\n"
            "Dear [EduLearn Contact Name],\n\n"
            "I am reaching out with a partnership proposal I believe holds genuine "
            "strategic value for EduLearn and its 2 million monthly active users.\n\n"
            "I would like to explore a content syndication arrangement between our "
            "platforms. The opportunity is straightforward: your users gain access to "
            "a curated stream of high-quality content that complements your existing "
            "offering, while both organisations benefit from expanded reach and shared "
            "audience engagement.\n\n"
            "To minimise commitment and build confidence in the collaboration, I propose "
            "we begin with a structured 3-month pilot. We are also pleased to propose "
            "a revenue sharing model structured 70/30 in EduLearn's favour, reflecting "
            "our confidence in the long-term mutual value this partnership will generate.\n\n"
            "I would welcome a 30-minute call to explore this further at a time that "
            "suits you. I am confident that with the right alignment, this could be "
            "the beginning of a highly productive relationship.\n\n"
            "Looking forward to your thoughts.\n\n"
            "Best regards,\n[Your Name]\n[Title] | [Company]"
        ),
    },
    {
        "id": 10,
        "intent": "Announce a new hybrid work policy to all staff",
        "key_facts": [
            "New policy takes effect January 6, 2025",
            "Requirement: minimum 3 days in the office per week",
            "Flexible days: Wednesday and Friday may be taken remotely",
            "Employees with approved medical accommodations are exempt",
            "All questions should be directed to the HR department",
        ],
        "tone": "Clear, professional, neutral",
        "reference_email": (
            "Subject: Important Update: New Hybrid Work Policy – Effective January 6, 2025\n\n"
            "Dear Team,\n\n"
            "I am writing to share an important update regarding our work arrangements, "
            "effective January 6, 2025.\n\n"
            "Going forward, all employees will be required to work from the office a "
            "minimum of three days per week. Wednesday and Friday will remain flexible "
            "remote working days, giving each of you the ability to plan your schedule "
            "accordingly.\n\n"
            "We recognise that individual circumstances vary. Employees with approved "
            "medical accommodations are exempt from this requirement, and existing "
            "accommodation arrangements will continue to be honoured without the need "
            "for further action.\n\n"
            "This policy reflects our commitment to maintaining a collaborative, "
            "connected workplace while preserving meaningful flexibility. We believe "
            "this balance supports both individual wellbeing and collective performance.\n\n"
            "Should you have any questions or need further clarification, please direct "
            "these to the HR department, who will be happy to assist.\n\n"
            "Thank you for your continued dedication.\n\n"
            "Warm regards,\n[Leadership / HR Team]"
        ),
    },
]