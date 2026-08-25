# Dataset Access Matrix

Last verified: 2026-08-25

This matrix covers the cohorts in the comparison table supplied to the project. Cohort sizes in papers and screenshots are release-specific; they are not reliable indicators of how many scans are downloadable today.

## Access tiers

- `DIRECT`: anonymous download with no account, agreement, or review.
- `REGISTER`: create an account and accept the applicable terms; no study-proposal review was identified.
- `APPLY`: submit a request, data-use agreement (DUA), or institutional certification before download.
- `PAUSED/PAID`: access has a fee or is not currently accepting new applications.

None of the cohorts below currently offers an anonymous `DIRECT` download of the complete individual-level imaging cohort.

## Cohort matrix

| Cohort | Tier | Current route | Practical status | NeuroClaw support |
| --- | --- | --- | --- | --- |
| ADNI | APPLY | [ADNI access through LONI IDA](https://adni.loni.usc.edu/data-samples/adni-data/) | DUA plus online application; approved users download from IDA. | `adni-skill` |
| ADNI-DOD | APPLY | [ADNI-DOD project instructions](https://adni.loni.usc.edu/support/experts-knowledge-base/question/?QID=920) | Use the `ADNIDOD` project after ADNI/IDA access; subject identifiers are `SCRNO`, not standard ADNI RIDs. | `adni-skill` |
| AIBL | APPLY | [AIBL data portal](https://data.aibl.org.au/adni/index.html) | LONI authentication and project access are required. | `aibl-skill` |
| BLSA (open) | APPLY | [NIA BLSA data use](https://www.nia.nih.gov/research/labs/blsa) and [BLSA open/permissioned listings](https://www.gaaindata.org/partner/BLSA) | "Open" is a sharing tier, not an unrestricted raw-MRI URL; obtain data through the BLSA/ADDI request route. | Catalogued only |
| HABS-HD | APPLY | [HABS-HD data requests](https://apps.unthsc.edu/itr/reports) | Proposal and DUA review; approved delivery is managed through LONI. | Catalogued only |
| MCSA | APPLY | [MCSA LONI DUA](https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=MCSA) | A curated de-identified imaging/clinical subset is broadly shared under a DUA; broader requests receive separate review. | Catalogued only |
| NIFD / FTLDNI | APPLY | [NIFD LONI DUA](https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=NIFD) | Controlled IDA download. The former OpenNeuro `ds004403` identifier is deleted and must not be used. | `nifd-skill` |
| U.S. POINTER | APPLY | [POINTER baseline data access](https://uspointer.net/dspDiscover.cfm) | Baseline data require a DUA and approval. Confirm that the approved package contains imaging before planning an imaging pipeline. | Catalogued only |
| PPMI | APPLY | [PPMI data access](https://www.ppmi-info.org/access-data-specimens/download-data) | DUA plus application; the site states that reviews are normally completed within one week. | `ppmi-skill` |
| SCAN | APPLY | [SCAN researcher access](https://scan.naccdata.org/) | Free request through NACC's Quick Access File system; defaced images, QC, summaries, and analysis variables are available after approval. | `scan-skill` |
| UK Biobank | PAUSED/PAID | [UK Biobank applications](https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/) and [fees](https://www.ukbiobank.ac.uk/use-our-data/fees/) | New applications are paused until late 2026. Imaging is a paid data tier and is normally analyzed in UKB-RAP. | `ukb-skill` (post-export) |
| WRAP | APPLY | [WRAP data requests](https://wrap.wisc.edu/data-requests-2/) | Project request and review; approved MRI/PET packages include NIfTI images and derived tables. | Catalogued only |
| HCP-YA | REGISTER | [HCP-YA 2025 release](https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release) | Register and accept HCP data-use terms, then download selected packages from ConnectomeDB powered by BALSA. | `hcpya-skill` |
| HCP-A / AABC | REGISTER | [AABC Release 2](https://www.humanconnectome.org/study/hcp-lifespan-aging/data-releases) | Register for BALSA, accept AABC terms, and use Aspera for imaging packages. Academic, nonprofit, or government email is required. | `hcpa-skill` |
| BIOCARD | APPLY | [BIOCARD resources for researchers](https://biocard.pathology.jhu.edu/resources-for-researchers/) | Submit a request; approved researchers receive credentials for data files and brain scans. | Catalogued only |
| ABCD | APPLY | [ABCD data access](https://docs.abcdstudy.org/latest/usage/access.html) | Release 7.0 is current; institutional NBDC DUC approval is required, with downloads through DEAP or the NBDC Data Access Platform. Releases 6.0+ are no longer distributed through NDA. | `abcd-skill` |
| EBDS | APPLY | [NDA Early Brain Development in Twins collection 2384](https://nda.nih.gov/edit_collection.html?id=2384) | Individual-level files require NDA/RAS access and an approved institutional request. Related EBDS collections may need to be queried separately. | Catalogued only |

## Recommended acquisition order

1. **SCAN**: best new addition for this repository. Access is free, defaced images are requestable, and the cohort links imaging to NACC clinical and cognitive data.
2. **HCP-A and HCP-YA**: lowest administrative friction. Start with a modality-specific package or a small subject subset because the full releases are very large.
3. **ADNI-DOD**: low implementation cost because it can reuse `adni-skill`; preserve `SCRNO` identifiers and keep the project separate from ordinary ADNI subjects.
4. **PPMI, AIBL, NIFD, and MCSA**: worthwhile after an IDA account and the corresponding DUA approvals are in place.
5. **BLSA, HABS-HD, WRAP, BIOCARD, POINTER, ABCD, and EBDS**: add full dataset adapters when an approved export is available and its real directory/schema contract can be tested.
6. **UK Biobank**: defer new acquisition while applications are paused; use `ukb-skill` only for an existing approved RAP project or a lawful local export.

## Operational rules

- Never store usernames, passwords, tokens, signed DUAs, or restricted subject manifests in the repository.
- Record the release name, access date, query/filter, subject count, modalities, checksum manifest, and governing DUA beside every local export.
- Treat portal approval and download as separate states. A dataset is not locally available until files and checksums have been verified.
- Do not redistribute controlled raw data or derived data when the governing agreement forbids it.
- Recheck the official link before each acquisition because access routes and terms change.
