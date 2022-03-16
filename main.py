'''
1. Estrarre dalla trascrizione presente nel file le cinque parole più usate nel contesto,
tramite un transformer: ti consiglio keybert, scegli un modello italiano che ti piace.
2. Elencare le 5 parole, i tempi in cui vengono dette, e le corrispondenti emozioni trovate
sul file rekognition.
Attenzione: i timecode di transcript e quello di rekognition sono diversi, devi adattarli
(e prendere il più vicino corrispondente) rekognition usa un intero per il timestamp,
transcribe un time, rekognition è rappresentato in millesimi.
Esempio 85.04 di transcribe = 85040 di rekognition

3. commenta lo script con le tue considerazioni, le scelte fatte e le tecniche usate.

4. finito e testato lo script inseriscilo su un tuo repository github e mandaci il link
'''


from keybert import KeyBERT
import json


def prova_mle():
    # Opening JSON file
    json_file = open('transcription.json')
    # Load json file
    json_dictionary = json.load(json_file)
    # Get transcript
    transcript = json_dictionary['results']['transcripts'][0]['transcript']
    print(transcript)

    # Get KeyBERT model, I choose paraphrase-multilingual-MiniLM-L12-v2 for multi-lingual documents (also italian)
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    # Extract 5 keywords by using KeyBERT model
    keywords = kw_model.extract_keywords(transcript)
    print('The most used keywords:\n', keywords)



    # Closing JSON file
    json_file.close()


if __name__ == "__main__":
    prova_mle()
