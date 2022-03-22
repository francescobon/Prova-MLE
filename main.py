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
import pandas as pd

'''Estrae dalla trascrizione presente nel file le cinque parole più usate nel contesto'''
def ex1 (transcription_df):

    # Get transcript
    transcript = transcription_df.loc['transcripts','results'][0]['transcript']
    # print(transcript)

    # Get KeyBERT model, I choose paraphrase-multilingual-MiniLM-L12-v2 for multi-lingual documents (also italian)
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    # Extract 5 keywords by using KeyBERT model
    keywords = kw_model.extract_keywords(transcript)

    return keywords


'''Stampa le 5 parole, i tempi in cui vengono dette, e le corrispondenti emozioni trovate
sul file rekognition'''
def ex2(keywords, transcription_df, rekognition_df):

    print("-- The words, times and emotions --\n")

    results = transcription_df.loc['items','results']

    # Search keywords in transcription results in order to get times and emotions from rekognition
    for (key, conf) in keywords:

        # Search the times and content in transcription
        for r in results:
            # Search if keys exist
            if 'start_time' in r and 'end_time' in r and 'alternatives' in r:
                start_time = float(r['start_time']) * 1000
                end_time = float(r['end_time']) * 1000
                content = r['alternatives'][0]['content'] #word to search

                # if the word is found then search in rekognition the times and emotions
                if content == key:
                    #print("is equal:", content)

                    #Search in rekognition the emotions and times
                    for index in range(len(rekognition_df)):
                        emotions = rekognition_df.loc[index, 'Face']['Emotions']
                        timestamp = rekognition_df.loc[index, 'Timestamp']

                        # time has to be contained in start and end times
                        if timestamp >= start_time and timestamp <= end_time:

                            #Print word, time, emotions
                            print ("Word:", content)
                            print ("Timestamp (ms):", timestamp)
                            print ("Start Time (ms):", start_time)
                            print ("End Time (ms):", end_time)
                            print ("Emotions:", [em['Type'] for em in emotions])
                            print ("\n---------\n")

def prova_mle():

    # Load json file
    transcription_json = pd.read_json('transcription.json')
    # print(json_dictionary)
    # Open Data Frame from json file
    transcription_df = pd.DataFrame(transcription_json)

    print ('---- Exercise 1 ----\n\n')
    keywords = ex1(transcription_df)
    print('The most used keywords:\n', [k for (k, c) in keywords])
    print ('\n--------------------\n\n\n')

    print ('---- Exercise 2 ----\n\n')
    # Load json file
    rekognition_json = pd.read_json('rekognition.json')
    # print(json_dictionary)
    # Open Data Frame from json file
    rekognition_df = pd.DataFrame(rekognition_json)

    ex2(keywords, transcription_df, rekognition_df)
    print ('\n--------------------\n\n\n')




if __name__ == "__main__":
    prova_mle()
