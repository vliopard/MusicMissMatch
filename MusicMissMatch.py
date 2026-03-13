#!/usr/bin/env python3
'''
Music Identifier using AcoustID + MusicBrainz + AudioTag + Shazam + AudD + SongFinder
Recognition chain (each is a fallback for the previous):
  1. AcoustID       - free, unlimited
  2. MusicBrainz    - free, direct lookup by AcoustID track ID
  3. AudioTag       - free up to 1000/month (own fingerprint engine)
  4. Shazam         - free, no key needed (via shazamio)
  5. AudD           - free up to 1000/month, permanent free tier (https://audd.io)
  6. SongFinder     - free tier via RapidAPI (last resort, strict rate limit)

Requirements:
    pip install pyacoustid requests
    Install fpcalc: https://acoustid.org/chromaprint
        - Ubuntu/Debian: sudo apt install libchromaprint-tools
        - macOS:         brew install chromaprint
        - Windows:       download from https://acoustid.org/chromaprint
'''

import sys
import re
import argparse
import time
import configparser
import platform
import unicodedata
import acoustid
import requests
from datetime import datetime
from pathlib import Path

if platform.system() == 'Windows':
    import msvcrt
else:
    import select
    import tty
    import termios

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# --- API Keys ---
# Keys are loaded from keys.ini in the same directory as this script.
# Copy keys.ini.example to keys.ini and fill in your values.
# keys.ini should never be committed to version control.
_INI_FILE = Path(__file__).parent / 'keys.ini'

def _load_ini():
    '''Load API keys from keys.ini using configparser.'''
    if not _INI_FILE.exists():
        print(f'[WARNING] keys.ini not found at {_INI_FILE}')
        print( '          Copy keys.ini.example to keys.ini and fill in your API keys.')
        return configparser.ConfigParser()
    cfg = configparser.ConfigParser()
    cfg.read(_INI_FILE, encoding='utf-8')
    return cfg

_cfg = _load_ini()

def _key(section, option):
    '''Safely read a key from the ini, returning empty string if missing.'''
    try:
        return _cfg.get(section, option).strip()
    except (configparser.NoSectionError, configparser.NoOptionError):
        return ''

def _setting(option, fallback):
    '''Read a value from [settings] section, casting to the type of fallback.'''
    try:
        raw = _cfg.get('settings', option).strip()
        return type(fallback)(raw)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return fallback
    except (ValueError, TypeError) as e:
        print(f'[WARNING] keys.ini [settings] {option} is invalid ({e}), using default: {fallback}')
        return fallback

ACOUSTID_KEY   = _key('acoustid',   'api_key')
AUDIOTAG_KEY   = _key('audiotag',   'api_key')
AUDD_KEY       = _key('audd',       'api_key')
SONGFINDER_KEY = _key('songfinder', 'api_key')

# --- API Endpoints ---
ACOUSTID_URL    = 'https://api.acoustid.org/v2/lookup'
MUSICBRAINZ_URL = 'https://musicbrainz.org/ws/2'
AUDIOTAG_URL    = 'https://audiotag.info/api'
AUDD_URL        = 'https://api.audd.io/'
SONGFINDER_URL  = 'https://songfinder-file-recognition.p.rapidapi.com/api/rapidapi/recognize/file'

# Characters not allowed in Windows filenames
WINDOWS_INVALID_CHARS = r'[<>:"/\\|?*\x00-\x1f]'

# Defaults — overridden by [settings] in keys.ini
SIMILARITY_THRESHOLD       = _setting('similarity_threshold',       70)
EARLY_STOP_THRESHOLD       = _setting('early_stop_threshold',       90)
CONFIRM_TIMEOUT            = _setting('confirm_timeout',            120)
# 'shorter' = prefer less bloated titles (default)
# 'longer'  = prefer more complete/detailed titles
TITLE_LENGTH_PREFERENCE    = _setting('title_length_preference',    'shorter')

# Engine priority for tiebreaking (lower = more trusted). Not user-configurable.
ENGINE_PRIORITY = {'AcoustID': 0, 'MusicBrainz': 1, 'AudioTag': 2, 'Shazam': 3, 'AudD': 4, 'SongFinder': 5}


def sanitize_filename(name):
    '''Remove or replace characters invalid in Windows filenames.'''
    name = re.sub(WINDOWS_INVALID_CHARS, '_', name)
    name = name.strip('. ')
    return name or 'Unknown'



def filename_similarity(original_stem, new_stem):
    '''
    Compare original filename (without extension) to proposed new name.
    Strips separators/punctuation and compares tokens using fuzzy matching.
    Returns a score from 0 (completely different) to 100 (identical).
    '''
    if not HAS_RAPIDFUZZ:
        return 100  # skip check if library not installed

    def normalize(s):
        # lowercase, replace separators and punctuation with spaces
        s = s.lower()
        s = re.sub(r'[-_&+,]', ' ', s)
        s = re.sub(r'[^a-z0-9 ]', ' ', s)
        s = re.sub(r' +', ' ', s).strip()
        return s

    orig = normalize(original_stem)
    new  = normalize(new_stem)

    # Use token_set_ratio: handles reordering and partial matches well
    return fuzz.token_set_ratio(orig, new)


def make_unique_path(path):
    '''If path exists, append (1), (2), etc. until unique.'''
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        new_path = parent / f'{stem} ({counter}){suffix}'
        if not new_path.exists():
            return new_path
        counter += 1


def make_result(title, artist, album, score, mb_id='', source=''):
    return {
        'title':  title  or 'Unknown',
        'artist': artist or 'Unknown',
        'album':  album  or 'Unknown',
        'score':  score,
        'mb_id':  mb_id,
        'source': source,
    }


# --- Engine 1: AcoustID ---
def try_acoustid(filepath):
    '''Fingerprint file and query AcoustID.'''
    try:
        duration, fingerprint = acoustid.fingerprint_file(str(filepath))
    except acoustid.FingerprintGenerationError as e:
        return None, None, f'Fingerprint error: {e}'

    post_data = {
        'client':      ACOUSTID_KEY,
        'duration':    int(duration),
        'fingerprint': fingerprint,
        'meta':        'recordings releasegroups compress',
    }
    try:
        resp = requests.post(ACOUSTID_URL, data=post_data, timeout=10)
        data = resp.json()
    except requests.RequestException as e:
        return None, None, f'AcoustID request failed: {e}'

    if data.get('status') != 'ok':
        err = data.get('error', {})
        return None, None, f'AcoustID error {err.get("code")}: {err.get("message")}'

    results = data.get('results', [])
    if not results:
        return None, None, 'AcoustID: no match'

    best = max(results, key=lambda r: r.get('score', 0))
    score = best.get('score', 0)
    acoustid_id = best.get('id', '')
    recordings = best.get('recordings', [])

    if not recordings:
        # Return the acoustid_id so MusicBrainz fallback can use it
        return None, acoustid_id, f'AcoustID: match ({score:.0%}) but no metadata'

    rec = recordings[0]
    title = rec.get('title', '')
    artists = rec.get('artists', [])
    artist = ', '.join(a.get('name', '') for a in artists)
    rgroups = rec.get('releasegroups', [])
    album = rgroups[0].get('title', '') if rgroups else ''
    mb_id = rec.get('id', '')

    return make_result(title, artist, album, score, mb_id, 'AcoustID'), acoustid_id, None


# --- Engine 2: MusicBrainz (via AcoustID track ID) ---
def try_musicbrainz(acoustid_id):
    '''Resolve AcoustID track ID -> MusicBrainz recording -> full metadata.'''
    if not acoustid_id:
        return None, 'MusicBrainz: no AcoustID track ID to look up'
    try:
        # Get MB recording IDs linked to this AcoustID
        resp = requests.get(
            ACOUSTID_URL,
            params={'client': ACOUSTID_KEY, 'trackid': acoustid_id, 'meta': 'recordings'},
            timeout=10,
        )
        data = resp.json()
        results = data.get('results', [])
        if not results:
            return None, 'MusicBrainz: no recordings linked to AcoustID'

        recordings = results[0].get('recordings', [])
        if not recordings:
            return None, 'MusicBrainz: AcoustID track has no linked recordings'

        mb_id = recordings[0].get('id', '')
        if not mb_id:
            return None, 'MusicBrainz: no MB recording ID found'

        # Full lookup on MusicBrainz
        mb_resp = requests.get(
            f'{MUSICBRAINZ_URL}/recording/{mb_id}',
            params={'fmt': 'json', 'inc': 'artists+releases+release-groups'},
            headers={'User-Agent': 'MusicIdentifier/1.0'},
            timeout=10,
        )
        mb = mb_resp.json()

        title = mb.get('title', '')
        ac = mb.get('artist-credit', [])
        artist = ''.join(
            (a.get('name') or a.get('artist', {}).get('name', '')) + a.get('joinphrase', '')
            for a in ac
        ).strip()
        releases = mb.get('releases', [])
        album = releases[0].get('title', '') if releases else ''

        return make_result(title, artist, album, 1.0, mb_id, 'MusicBrainz'), None

    except Exception as e:
        return None, f'MusicBrainz error: {e}'


# --- Engine 3: AudioTag ---
def try_audiotag(filepath):
    '''Upload file to AudioTag and poll for result.'''
    if not AUDIOTAG_KEY or AUDIOTAG_KEY == 'YOUR_AUDIOTAG_KEY':
        return None, 'AudioTag: no API key configured'
    try:
        # Step 1: upload
        with open(filepath, 'rb') as f:
            resp = requests.post(
                AUDIOTAG_URL,
                data={'apikey': AUDIOTAG_KEY, 'action': 'identify'},
                files={'file': f},
                timeout=30,
            )
        data = resp.json()
        if data.get('status') != 'success':
            return None, f'AudioTag upload failed: {data}'

        token = data.get('token')
        if not token:
            return None, 'AudioTag: no token returned'

        # Step 2: poll for result (up to 10 attempts)
        for _ in range(10):
            time.sleep(2)
            poll = requests.post(
                AUDIOTAG_URL,
                data={'apikey': AUDIOTAG_KEY, 'action': 'get_result', 'token': token},
                timeout=10,
            )
            result = poll.json()
            if result.get('status') == 'success' and result.get('found'):
                tracks = result.get('tracks', [{}])
                track = tracks[0] if tracks else {}
                title  = track.get('title', '')
                artist = track.get('artist', '')
                album  = track.get('album', '')
                return make_result(title, artist, album, 1.0, '', 'AudioTag'), None
            if result.get('status') == 'success' and not result.get('found'):
                return None, 'AudioTag: no match found'

        return None, 'AudioTag: timed out waiting for result'

    except Exception as e:
        return None, f'AudioTag error: {e}'



# --- Engine 4: Shazam ---
def try_shazam(filepath):
    '''Identify track using Shazam engine via shazamio (free, no key needed).'''
    try:
        import asyncio
        from shazamio import Shazam

        async def _recognize():
            shazam = Shazam()
            return await shazam.recognize(str(filepath))

        data = asyncio.run(_recognize())

        track = data.get('track')
        if not track:
            return None, 'Shazam: no match found'

        title  = track.get('title', '')
        artist = track.get('subtitle', '')  # shazamio uses subtitle for artist
        sections = track.get('sections', [])
        album = ''
        for section in sections:
            for meta in section.get('metadata', []):
                if meta.get('title', '').lower() == 'album':
                    album = meta.get('text', '')
                    break

        return make_result(title, artist, album, 1.0, '', 'Shazam'), None

    except ImportError:
        return None, 'Shazam: shazamio not installed (pip install shazamio)'
    except Exception as e:
        return None, f'Shazam error: {e}'


# --- Engine 5: AudD ---
def try_audd(filepath):
    '''Upload file to AudD music recognition API.'''
    if not AUDD_KEY or AUDD_KEY == 'AUDD_KEY':
        return None, 'AudD: no API key configured'
    try:
        with open(filepath, 'rb') as f:
            resp = requests.post(
                AUDD_URL,
                data={
                    'api_token': AUDD_KEY,
                    'return': 'spotify,apple_music',
                },
                files={'file': f},
                timeout=30,
            )
        data = resp.json()
        if data.get('status') != 'success':
            return None, f'AudD error: {data}'
        result = data.get('result')
        if not result:
            return None, 'AudD: no match found'
        title  = result.get('title', '')
        artist = result.get('artist', '')
        # Normalize featured artists (e.g. "Artist (feat. Other)") for cleaner filenames
        artist = re.sub(r'\s*\(feat\..*?\)', '', artist, flags=re.I).strip()
        album  = result.get('album', '')
        return make_result(title, artist, album, 1.0, '', 'AudD'), None
    except Exception as e:
        return None, f'AudD error: {e}'


# --- Engine 6: SongFinder ---
SONGFINDER_MIN_INTERVAL = _setting('songfinder_min_interval', 60)  # minimum seconds between SongFinder calls
SONGFINDER_MAX_RETRIES  = 3

_songfinder_last_call = 0.0  # timestamp of the last successful SongFinder request

def try_songfinder(filepath):
    '''Upload file to SongFinder via RapidAPI, respecting the 1 req/min rate limit
    by tracking the last call time and waiting proactively before each request.'''
    global _songfinder_last_call

    if not SONGFINDER_KEY or SONGFINDER_KEY == 'YOUR_SONGFINDER_KEY':
        return None, 'SongFinder: no API key configured'

    # Wait proactively so we never hit the rate limit in the first place
    elapsed = time.time() - _songfinder_last_call
    if elapsed < SONGFINDER_MIN_INTERVAL:
        time.sleep(SONGFINDER_MIN_INTERVAL - elapsed)

    try:
        with open(filepath, 'rb') as f:
            resp = requests.post(
                SONGFINDER_URL,
                headers={
                    'x-rapidapi-host': 'songfinder-file-recognition.p.rapidapi.com',
                    'x-rapidapi-key':  SONGFINDER_KEY,
                },
                files={'file': (filepath.name, f, 'audio/mpeg')},
                timeout=30,
            )

        _songfinder_last_call = time.time()

        if resp.status_code == 429:
            # Shouldn't happen with proactive waiting, but handle gracefully
            retry_after = int(resp.headers.get('Retry-After', SONGFINDER_MIN_INTERVAL))
            time.sleep(retry_after)
            return None, 'SongFinder: unexpected rate limit, skipping'

        data = resp.json()

        if not data.get('success'):
            msg = data.get('message', 'unknown')
            if 'rate limit' in msg.lower() or 'exceeded' in msg.lower():
                time.sleep(SONGFINDER_MIN_INTERVAL)
                return None, 'SongFinder: unexpected rate limit, skipping'
            return None, f'SongFinder error: {msg}'

        if data.get('noMatch'):
            return None, 'SongFinder: no match found'

        track  = data.get('track', {})
        title  = track.get('title', '')
        artist = track.get('artist', '')
        album  = track.get('album', '')
        return make_result(title, artist, album, 1.0, '', 'SongFinder'), None

    except Exception as e:
        return None, f'SongFinder error: {e}'


# --- Main identification pipeline ---
def identify_track(filepath):
    '''
    Generator — yields one (result, errors_so_far) tuple per engine that finds a match.
    The caller decides whether to accept a candidate or advance to the next engine.
    Engines that find nothing are skipped silently; their errors accumulate in the list.
    When all engines are exhausted the generator ends; the caller can check the errors
    list it received on the last yield (or use the empty-candidate path) to report failure.
    '''
    errors = []

    # 1. AcoustID
    result, acoustid_id, err = try_acoustid(filepath)
    if result:
        yield result, errors
    else:
        errors.append(f'AcoustID: {err}')

    # 2. MusicBrainz (uses the AcoustID track ID from step 1)
    result, err = try_musicbrainz(acoustid_id)
    if result:
        yield result, errors
    else:
        errors.append(f'MusicBrainz: {err}')

    # 3. AudioTag
    result, err = try_audiotag(filepath)
    if result:
        yield result, errors
    else:
        errors.append(f'AudioTag: {err}')

    # 4. Shazam
    result, err = try_shazam(filepath)
    if result:
        yield result, errors
    else:
        errors.append(f'Shazam: {err}')

    # 5. AudD
    result, err = try_audd(filepath)
    if result:
        yield result, errors
    else:
        errors.append(f'AudD: {err}')

    # 6. SongFinder
    result, err = try_songfinder(filepath)
    if result:
        yield result, errors
    else:
        errors.append(f'SongFinder: {err}')


# --- Timed input (cross-platform) ---
def timed_input(prompt, timeout=CONFIRM_TIMEOUT, default='n'):
    '''Display prompt and read a line from stdin. If the user does not press
    Enter within `timeout` seconds, return `default` and print a notice.
    Works on Windows (msvcrt) and Linux/Mac (select + tty).'''
    print(prompt, end='', flush=True)
    start = time.time()
    chars = []

    if platform.system() == 'Windows':
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwche()
                if ch in ('\r', '\n'):
                    print()
                    return (''.join(chars).strip().lower()) or default
                elif ch == '\x08':         # Backspace
                    if chars:
                        chars.pop()
                        sys.stdout.write(' \b')
                        sys.stdout.flush()
                elif ch == '\x03':         # Ctrl+C
                    raise KeyboardInterrupt
                else:
                    chars.append(ch)
            elif time.time() - start >= timeout:
                print(f'\n  (no answer after {timeout}s, auto-skipping)')
                return default
            else:
                time.sleep(0.05)
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch in ('\r', '\n'):
                        print()
                        return (''.join(chars).strip().lower()) or default
                    elif ch == '\x7f':     # Backspace (DEL on Unix)
                        if chars:
                            chars.pop()
                            sys.stdout.write('\b \b')
                            sys.stdout.flush()
                    elif ch == '\x03':     # Ctrl+C
                        raise KeyboardInterrupt
                    else:
                        chars.append(ch)
                        sys.stdout.write(ch)
                        sys.stdout.flush()
                elif time.time() - start >= timeout:
                    print(f'\n  (no answer after {timeout}s, auto-skipping)')
                    return default
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# --- Rename helper (shared by auto-rename and manual paths) ---
def _do_rename(filepath, result, log, similarity):
    '''Perform the actual rename and write the log entry. Returns new path.'''
    artist   = sanitize_filename(result['artist'])
    album    = sanitize_filename(result['album'])
    title    = sanitize_filename(result['title'])
    new_name = f'{artist} -- {album} -- {title}.mp3'
    new_path = make_unique_path(filepath.parent / new_name)
    filepath.rename(new_path)
    print(f'  Renamed: [{new_path.name}]\n')
    log.write(f'RENAMED:  {new_path.name}\n')
    log.write(f'SOURCE:   {result["source"]}\n')
    log.write(f'TITLE:    {result["title"]}\n')
    log.write(f'ARTIST:   {result["artist"]}\n')
    log.write(f'ALBUM:    {result["album"]}\n')
    log.write(f'SCORE:    {result["score"]:.0%}\n')
    log.write(f'SIMILARITY: {similarity:.0f}%\n')
    if result['mb_id']:
        log.write(f'MB ID:    https://musicbrainz.org/recording/{result["mb_id"]}\n')
    log.write('\n')
    return new_path


# --- Directory processor ---
def process_directory(directory, mode='stop_when_very_hi'):
    directory = Path(directory)
    mp3_files = sorted(directory.glob('*.mp3'))

    if not mp3_files:
        print(f'No MP3 files found in: {directory}')
        return

    log_path = directory / 'identify_music.log'
    total = len(mp3_files)
    print(f'Found {total} MP3 file(s) in: {directory}')
    print(f'Log: {log_path}')
    print()

    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f'=== Run: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | mode: {mode} ===\n\n')

        for i, filepath in enumerate(mp3_files, 1):
            print(f'[{i}/{total}] {filepath.name}')
            log.write(f'ORIGINAL: {filepath.name}\n')

            renamed    = False
            candidates = []   # (result, similarity, new_path) per engine that matched
            all_errors = []

            for result, all_errors in identify_track(filepath):
                artist     = sanitize_filename(result['artist'])
                album      = sanitize_filename(result['album'])
                title      = sanitize_filename(result['title'])
                new_name   = f'{artist} -- {album} -- {title}.mp3'
                new_path   = make_unique_path(filepath.parent / new_name)
                similarity = filename_similarity(filepath.stem, new_path.stem)

                # Always print a compact one-liner so the user can follow progress;
                # full detail is printed later only for the winning candidate.
                print(f'  {result["source"]:14s} [{similarity:3.0f}%]  [{result["artist"]} — {result["album"]} — {result["title"]}]')

                candidates.append((result, similarity, new_path))

                if mode == 'first_win':
                    break  # take the first result regardless of score

                elif mode == 'stop_when_very_hi' and similarity >= EARLY_STOP_THRESHOLD:
                    print(f'  Early stop ({similarity:.0f}% >= {EARLY_STOP_THRESHOLD}%).\n')
                    break
                # high_similarity and stop_when_very_hi (below threshold): keep going

            # --- Post-loop decision ---
            if not candidates:
                print('  ERROR: no engine could identify this file.\n')
                log.write('RENAMED:  (not renamed)\n')
                log.write(f'ERROR:    {" | ".join(all_errors)}\n')
                log.write('\n')
            else:
                # Pick the highest-similarity candidate
                # Sort key: (similarity, consensus_count, metadata_length) — all descending.
                # Consensus: group by exact title+artist+album match across engines.
                # Keys are pre-computed from the unmodified list so that sorting does
                # not mutate candidates while consensus_count is still reading it.
                def normalize(s):
                    '''Normalize unicode for comparison: replace typographic punctuation,
                    decompose accents, fold case, drop remaining non-ascii.'''
                    s = s.replace('‘', "'").replace('’', "'")  # curly apostrophes
                    s = s.replace('“', '"').replace('”', '"')  # curly quotes
                    s = s.replace('—', '-').replace('–', '-')  # em/en dash
                    s = unicodedata.normalize('NFKD', s)
                    s = s.encode('ascii', 'ignore').decode('ascii')
                    return s.lower().strip()

                def make_sort_key(candidate, _candidates=candidates):
                    r = candidate[0]
                    grp = (normalize(r['title']), normalize(r['artist']), normalize(r['album']))
                    consensus = sum(
                        1 for other, _, _ in _candidates
                        if normalize(other['title'])  == grp[0]
                        and normalize(other['artist']) == grp[1]
                        and normalize(other['album'])  == grp[2]
                    )
                    return (
                        candidate[1],                                    # 1. similarity (higher=better)
                        consensus,                                        # 2. consensus  (higher=better)
                        -ENGINE_PRIORITY.get(r['source'], 99),            # 3. engine priority (lower index=better)
                        len(r['title']) if TITLE_LENGTH_PREFERENCE == 'longer' else -len(r['title']),  # 4. title length
                    )

                sort_keys = {id(c): make_sort_key(c) for c in candidates}
                candidates.sort(key=lambda x: sort_keys[id(x)], reverse=True)
                # Log all candidates so the full picture is available for review
                log.write('CANDIDATES:\n')
                for c_result, c_sim, _ in candidates:
                    log.write(f'  {c_result["source"]:14s} {c_sim:3.0f}%  {c_result["artist"]} — {c_result["album"]} — {c_result["title"]}\n')
                result, similarity, new_path = candidates[0]
                others = candidates[1:]

                # Print full detail for the winner
                print(f'\n  Winner:  {result["source"]} ({similarity:.0f}%)')
                print(f'  Title:   [{result["title"]}]')
                print(f'  Artist:  [{result["artist"]}]')
                print(f'  Album:   [{result["album"]}]')
                print(f'  Score:   [{result["score"]:.0%}]')
                if result['mb_id']:
                    print(f'  MB ID:   https://musicbrainz.org/recording/{result["mb_id"]}')
                if others:
                    print(f'  Others:  {', '.join(f'{r["source"]} ({s:.0f}%)' for r, s, _ in others)}')

                if similarity >= SIMILARITY_THRESHOLD:
                    # Confident enough — auto-rename, no prompt
                    _do_rename(filepath, result, log, similarity)
                else:
                    # Below threshold — ask once
                    print(f'  Rename to: [{new_path.name}]')
                    answer = timed_input(f'  Confirm? [y=yes / n=skip / m=manual] (auto-skip in {CONFIRM_TIMEOUT}s): ')

                    if answer == 'y':
                        _do_rename(filepath, result, log, similarity)
                    elif answer == 'm':
                        m_artist = input(f'  Artist [{result["artist"]}]: ').strip() or result['artist']
                        m_album  = input(f'  Album  [{result["album"]}]: ').strip() or result['album']
                        m_title  = input(f'  Title  [{result["title"]}]: ').strip() or result['title']
                        result.update({'artist': m_artist, 'album': m_album, 'title': m_title})
                        _do_rename(filepath, result, log, similarity)
                    else:
                        print('  Skipped. File NOT renamed.')
                        log.write('RENAMED:  (skipped by user)\n')
                        log.write(f'BEST:     {result["source"]} {similarity:.0f}% -> {new_path.name}\n')
                        log.write('\n')
                    print()

        log.write('=== End of run ===\n\n')

    print(f'Done! Log saved to: {log_path}')


def validate_keys():
    '''Test all configured API keys before processing any files.'''
    print('Validating API keys...')
    ok = True

    # --- AcoustID (required) ---
    if ACOUSTID_KEY == 'YOUR_ACOUSTID_KEY':
        print('  [MISSING]  AcoustID  -> set ACOUSTID_KEY  (https://acoustid.org/login)')
        return False  # hard stop, no point continuing
    try:
        resp = requests.post(
            ACOUSTID_URL,
            data={'client': ACOUSTID_KEY, 'duration': 1, 'fingerprint': 'test', 'meta': ''},
            timeout=10,
        )
        data = resp.json()
        err_code = data.get('error', {}).get('code')
        if err_code == 4:
            print('  [INVALID]  AcoustID  -> key rejected (https://acoustid.org/login)')
            return False  # hard stop
        # any other error (e.g. bad fingerprint) means the key itself was accepted
        print('  [OK]       AcoustID')
    except Exception as e:
        print(f'  [ERROR]    AcoustID  -> could not reach API: {e}')
        return False

    # --- AudioTag (optional) ---
    global AUDIOTAG_KEY
    if AUDIOTAG_KEY == 'YOUR_AUDIOTAG_KEY':
        print('  [SKIPPED]  AudioTag   -> no key set (engine disabled)')
    else:
        try:
            resp = requests.post(
                AUDIOTAG_URL,
                data={'apikey': AUDIOTAG_KEY, 'action': 'get_result', 'token': 'test'},
                timeout=10,
            )
            data = resp.json()
            if 'invalid' in str(data).lower() or 'auth' in str(data).lower() or 'key' in str(data).lower():
                print('  [INVALID]  AudioTag   -> key rejected, engine disabled')
                AUDIOTAG_KEY = None
            else:
                print('  [OK]       AudioTag')
        except Exception as e:
            print(f'  [ERROR]    AudioTag   -> could not reach API, engine disabled: {e}')
            AUDIOTAG_KEY = None

    # --- AudD (optional) ---
    global AUDD_KEY
    if AUDD_KEY == 'AUDD_KEY':
        print('  [SKIPPED]  AudD       -> no key set (engine disabled)')
    else:
        try:
            resp = requests.post(
                AUDD_URL,
                data={'api_token': AUDD_KEY},
                timeout=10,
            )
            data = resp.json()
            # AudD returns error code 901 for wrong token
            if data.get('error', {}).get('error_code') == 901:
                print('  [INVALID]  AudD       -> key rejected, engine disabled')
                AUDD_KEY = None
            else:
                print('  [OK]       AudD')
        except Exception as e:
            print(f'  [ERROR]    AudD       -> could not reach API, engine disabled: {e}')
            AUDD_KEY = None

    # --- SongFinder (optional) ---
    global SONGFINDER_KEY
    if SONGFINDER_KEY == 'YOUR_SONGFINDER_KEY':
        print('  [SKIPPED]  SongFinder -> no key set (engine disabled)')
    else:
        try:
            resp = requests.get(
                'https://songfinder-file-recognition.p.rapidapi.com/',
                headers={
                    'x-rapidapi-host': 'songfinder-file-recognition.p.rapidapi.com',
                    'x-rapidapi-key':  SONGFINDER_KEY,
                },
                timeout=10,
            )
            if resp.status_code == 403:
                print('  [INVALID]  SongFinder -> key rejected, engine disabled')
                SONGFINDER_KEY = None
            else:
                print('  [OK]       SongFinder')
        except Exception as e:
            print(f'  [ERROR]    SongFinder -> could not reach API, engine disabled: {e}')
            SONGFINDER_KEY = None

    return ok


MODES = ['first_win', 'high_similarity', 'stop_when_very_hi']

def main():
    parser = argparse.ArgumentParser(
        description='Identify and rename MP3 files using multiple music recognition engines.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''modes:
  first_win          rename on the first engine that returns any result (fastest)
  high_similarity    run all engines, pick the one with highest similarity
  stop_when_very_hi  run all engines, but stop early if similarity >= EARLY_STOP_THRESHOLD
                     (default: 90%%) — best balance of speed and accuracy
''')
    parser.add_argument('directory', help='directory containing MP3 files to rename')
    parser.add_argument(
        '--mode', choices=MODES, default='stop_when_very_hi',
        help='engine selection strategy (default: stop_when_very_hi)'
    )
    args = parser.parse_args()

    if not validate_keys():
        print('\nFix the key issues above before running.')
        sys.exit(1)

    print(f'\nMode: {args.mode}\n')
    process_directory(args.directory, mode=args.mode)


if __name__ == '__main__':
    main()
