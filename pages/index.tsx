import type { NextPage } from 'next';
import Head from 'next/head';
import styles from '../styles/Home.module.css';
import ImageCanvas from "../components/ImageCanvas";


const Home: NextPage = () => {  
  
  return (
    <div className={styles.container}>
      <Head>
        <title>photo2levitan web demo</title>
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>
          photo2levitan web demo
        </h1>

      <span className={styles.grid}>Загрузите изображение или вставьте ссылку на него. Также доступен <a href="https://t.me/photo2levitan_bot" target="_blank">Телеграм бот</a></span>

      <ImageCanvas width={256} height={256}/>
      <div id="result" className="mt-3">
      </div>
      </main>
    </div>
  )
}

export default Home
