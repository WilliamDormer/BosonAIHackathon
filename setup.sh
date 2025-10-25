#!/bin/bash

# Read secret_keys.txt and pull out the username (first line), and key (second line.)


# Set the environment variable for BOSON_API_KEY
USERNAME=$(head -n 1 src/config/secret_keys.txt)
KEY=$(tail -n 1 src/config/secret_keys.txt)

echo $USERNAME
echo $KEY

export BOSON_USERNAME=$USERNAME
export BOSON_API_KEY=$KEY
