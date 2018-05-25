#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the additional dependency `pyaudio`. To install
using pip:

    pip install pyaudio

Example usage:
    python transcribe_streaming_mic.py
"""

# [START import_libraries]
from __future__ import division

import re
import sys

import random
import subprocess

from gtts import gTTS
import os

import psycopg2 #for connecting to postgres database

from word2number import w2n #currently the STT will output "four" and it won't align with the database's "4"
from num2words import num2words #might be easier to just convert the db recognition bank instead of every STT utterance

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
# [END import_libraries]

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

utterances = ["sure", "okay...", "great...", "alright, yeah...", "go on...", "got it", "I see..."]
last_recorded_phrase = ""
utterance_choice = ""

#connect to database
conn = psycopg2.connect("dbname=cars user=sydneyzink")
cur = conn.cursor()

#for now, just collect all words in the database single table for recognition
cur.execute("SELECT * FROM cars_table;")
lists_of_tableline_words = cur.fetchall()
list_all_words = []
for list_of_words in lists_of_tableline_words:
    list_all_words += list_of_words
list_all_wordstrings = [str(i).lower() for i in list_all_words]
list_all_wordstrings = list(set(list_all_wordstrings))
#db_integers = [i for i in list_all_wordstrings if isinstance(i, int)]
bad_num2word_integerstring = " point zero" #num2words attaches 'point zero' to integers....k. Dumb.
for term_ind in range(0, len(list_all_wordstrings)):
    try: #found an integer in the db values that we want to turn into words instead
        int(list_all_wordstrings[term_ind])
        num2words_rep = num2words(list_all_wordstrings[term_ind]).encode("utf-8")
        list_all_wordstrings[term_ind] = num2words_rep[:num2words_rep.find(bad_num2word_integerstring)]
    except ValueError:
        continue
print (list_all_wordstrings)

#put in some dummy words for now so I can keep using the same audio clip instead of car talk
#list_all_wordstrings = ["maui", "president", "sanctuary", "island"]
#print (list_all_wordstrings)

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)
# [END audio_stream]


def listen_print_loop(responses, filter_words):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    aggregated_filter_words = filter_words

    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript
        last_recorded_phrase = ((transcript.lower()).encode("utf-8")).split()
        print (last_recorded_phrase) #last utterance picked up, represented as a list term by term.
        #print (set(list_all_wordstrings)) #all table keywords

        std_resp = ""
        if (len(set(last_recorded_phrase).intersection(set(list_all_wordstrings)))): #reelvant db words in utterance
            overlap = set(last_recorded_phrase).intersection(set(list_all_wordstrings))
            overlap = list(overlap)
            print (overlap)
            dbselect = conn.cursor();
            anded_terms = ''
            for i in range(0, len(overlap)):
                if (i is 0):
                    anded_terms += "\'"
                anded_terms += overlap[i]
                if (i != len(overlap)-1):
                    anded_terms += ' & '
                else:
                    anded_terms += "\'"
            print ("SEARCHING CAR_TABLE VALUES FOR: " + anded_terms)
            if (aggregated_filter_words != ''):
                aggregated_filter_words = aggregated_filter_words[:-1] + ' & ' + anded_terms[1:]
            else:
                aggregated_filter_words = anded_terms
            query_text = "SELECT * from cars_table where to_tsvector(cars_table::text) @@ to_tsquery(" + aggregated_filter_words + ");"
            dbselect.execute(query_text)
            filtered_res = dbselect.fetchall()
            print (filtered_res)

            #std_resp = "Sure,"
            ind = 0
            while (ind < len(overlap)):
                std_resp = std_resp + " " + str(overlap[ind]) + " "
                ind += 1
                # the following puts "and" between recognized terms if there are multiple, but that sounds worse than just listing them
                # if (ind < len(overlap)):
                #     std_resp += "and "
                #else:
                #    std_resp += ", "
                std_resp += ", "
            #print("MOST RECENT UTTERANCE FEEDBACK TERMS IN BACKCHANNEL: " + std_resp)



        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)
            break

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0

    return (std_resp, aggregated_filter_words)

def keyword_repeat(text):
    print (transcript)

def say(text_resp):
    #subprocess.call(['say', text_resp])
    tts_response = gTTS(text=text_resp, lang='en')
    tts_response.save("resp_file.mp3")
    os.system("play resp_file.mp3")

def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'en-US'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        single_utterance=True,
        interim_results=False)
    i = 0
    filtering_words = ''
    while (i < 7):
        with MicrophoneStream(RATE, CHUNK) as stream:
            utterance_choice = ""
            print("in stream")
            audio_generator = stream.generator()
            requests = (types.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use.
            #while(True):
            (prepend, filtering_words) = listen_print_loop(responses, filtering_words)
            print ("prepend to backchannel: " + prepend)
            utterance_index = random.randint(0, len(utterances)-1)
            utterance_choice += (" " + prepend + utterances[utterance_index])
            say(utterance_choice)
        i+=1


if __name__ == '__main__':
    main()
