// Classification and mode-gating of files dragged onto the app window.
//
// Dropped-file paths come from Electron's File.path, so the renderer sees
// raw OS paths — Windows ("C:\\Users\\x\\clip.MP4") or POSIX
// ("/Users/x/clip.mp4") style. Extension parsing must handle both
// separators and be case-insensitive.
//
// Which files a mode accepts is defined by the active Mode selector and
// mirrors the DEFAULT (first) Browse File dialog filter for that mode in
// main/browseFileFilters.ts: face/hand take images + videos, speech takes
// audio + video. CRA's module scope keeps src/ from importing main/, so the
// lists are duplicated here; tests/drag-drop.spec.ts asserts they stay in
// sync with the dialog filters.

export type DroppedFileKind = 'video' | 'image' | 'audio' | 'unsupported';

const IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'];
const VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv'];
const AUDIO_EXTENSIONS = ['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg'];

// Per-mode accepted extensions == buildBrowseFileFilters(mode)[0].extensions.
const IMAGE_VIDEO_MODE = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'mp4', 'avi', 'mov'];
const AUDIO_VIDEO_MODE = ['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'mp4', 'mov', 'mkv'];
const MODE_ACCEPTED_EXTENSIONS: Record<string, ReadonlyArray<string>> = {
    face: IMAGE_VIDEO_MODE,
    hand: IMAGE_VIDEO_MODE,
    speech: AUDIO_VIDEO_MODE,
};

export function getPathExtension(filePath: string): string {
    const basename = filePath.split(/[\\/]/).pop() ?? '';
    const dot = basename.lastIndexOf('.');
    // dot <= 0 covers "no extension" and dotfiles like ".DS_Store".
    if (dot <= 0) return '';
    return basename.slice(dot + 1).toLowerCase();
}

export function classifyDroppedPath(filePath: string): DroppedFileKind {
    const extension = getPathExtension(filePath);
    if (VIDEO_EXTENSIONS.includes(extension)) return 'video';
    if (IMAGE_EXTENSIONS.includes(extension)) return 'image';
    if (AUDIO_EXTENSIONS.includes(extension)) return 'audio';
    return 'unsupported';
}

export function allowedExtensionsForMode(mode?: string): ReadonlyArray<string> {
    return MODE_ACCEPTED_EXTENSIONS[mode ?? ''] ?? IMAGE_VIDEO_MODE;
}

export function isDropAllowedForMode(filePath: string, mode?: string): boolean {
    return allowedExtensionsForMode(mode).includes(getPathExtension(filePath));
}
