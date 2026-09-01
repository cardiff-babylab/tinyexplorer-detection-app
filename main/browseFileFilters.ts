// Filters for the single-file "Browse File" dialog, extracted from the
// browse-file IPC handler so filter ordering is unit-testable.
//
// Ordering matters on macOS: Electron's open dialog shows a file-type
// dropdown that defaults to the FIRST filter, and files outside the active
// filter are greyed out. Every mode accepts video input, so the first filter
// of every mode must include the video extensions.

export interface FileFilter {
    name: string;
    extensions: string[];
}

export function buildBrowseFileFilters(mode?: string): FileFilter[] {
    const speechFilters = mode === "speech";
    const imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'];
    const videoExtensions = ['mp4', 'avi', 'mov'];
    const audioVideoFilter = {
        name: 'Audio and Video',
        extensions: ['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'mp4', 'mov', 'mkv'],
    };
    return speechFilters
        ? [
            audioVideoFilter,
            { name: 'All Files', extensions: ['*'] },
        ]
        : [
            { name: 'Images and Videos', extensions: [...imageExtensions, ...videoExtensions] },
            { name: 'Images', extensions: imageExtensions },
            { name: 'Videos', extensions: videoExtensions },
            audioVideoFilter,
            { name: 'All Files', extensions: ['*'] },
        ];
}
