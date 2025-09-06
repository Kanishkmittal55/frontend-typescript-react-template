// Global Declaration - Any .d.ts file in your project is automatically picked up 
// by typescript , if it is within the include paths defined in tsconfig.json

declare module 'pdfjs-dist/build/pdf' {
    const pdfjs: any;
    export = pdfjs;
  }
  
  declare module 'pdfjs-dist/build/pdf.worker.entry' {
    const pdfjsWorker: any;
    export = pdfjsWorker;
  }
  