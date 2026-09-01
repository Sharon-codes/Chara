import '@/styles/globals.css';
import type { AppProps } from 'next/app';
import Head from 'next/head';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>Chara: Molecular-Dynamics-Guided Survival Generalization</title>
        <meta name="description" content="Next.js Interactive Presentation Deck for Chara Survival Platform" />
        <link rel="icon" href="/iit-mandi-logo.png" />
      </Head>
      <Component {...pageProps} />
    </>
  );
}
