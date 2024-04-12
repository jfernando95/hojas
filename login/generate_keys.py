import pickle
from pathlib import Path
import yaml
import streamlit_authenticator as stauth

# Generar las contraseñas hasheadas
hashed_passwords = stauth.Hasher(['abc', 'def']).generate()

# Crear la estructura completa de credenciales con contraseñas hasheadas
credentials = {
    'cookie': {
        'expiry_days': 30,
        'key': 'some_signature_key',
        'name': 'some_cookie_name'
    },
    'credentials': {
        'usernames': {
            'dbaldwin': {
                'email': 'dbaldwin@gmail.com',
                'name': 'David Baldwin',
                'password': hashed_passwords[0]
            },
            'jsmith': {
                'email': 'jsmith@gmail.com',
                'name': 'John Smith',
                'password': hashed_passwords[1]
            },
        },
    },
    'preauthorized': {
        'emails': ['melsby@gmail.com']
    }
}

# Guardar la estructura completa de credenciales en un archivo YAML
with open('credentials.yaml', 'w') as yaml_file:
    yaml.dump(credentials, yaml_file, default_flow_style=False)