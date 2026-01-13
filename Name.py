import PyPDF2
import warnings
import logging

warnings.filterwarnings('ignore', category=UserWarning, module='PyPDF2')
logging.getLogger('PyPDF2').setLevel(logging.ERROR)

pdf_file_1 = open('D:\AppleProject\Knowledge\Brown rot_ Causes, Symptoms &amp; Control _ RHS Advice.pdf (1).pdf','rb')
pdf_file_2 = open('D:\AppleProject\Knowledge\R. K. Prasad, et al (1).pdf','rb')
pdf_file_3 = open('D:\AppleProject\Knowledge\Apple scab _ Pome fruits _ Fruit and nut diseases _ Plant diseases _ Biosecurity _ Agriculture Victoria (1).pdf','rb')

pdf_read_1 = PyPDF2.PdfReader(pdf_file_1)
fulltext_1 = ""
for page in pdf_read_1.pages:
  fulltext_1 += page.extract_text()

pdf_read_2 =PyPDF2.PdfReader(pdf_file_2)
fulltext_2 =""
for page in pdf_read_2.pages:
  fulltext_2 += page.extract_text()

pdf_read_3 = PyPDF2.PdfReader(pdf_file_3)
fulltext_3 = ""
for page in pdf_read_3.pages:
  fulltext_3 += page.extract_text()

fulltext_2 = fulltext_2[4270:21580]
fulltext_3 = fulltext_3[183:11600]
fulltext_1 = fulltext_1[75:]