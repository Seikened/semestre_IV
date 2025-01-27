import { motion } from "framer-motion";
import { TimelinePoint } from "./timeline-point";

const timelineData = [
  {
    year: 1963,
    title: "Sketchpad",
    description: "Ivan Sutherland desarrolló el primer programa interactivo con gráficos, llamado Sketchpad. Permitía a los usuarios dibujar directamente en la pantalla con un lápiz óptico, manipulando figuras geométricas en tiempo real.",
    image:"https://tamarind.unm.edu/wp-content/uploads/Ivan_Sutherland1962.jpg"
  },
  {
    year: 1973,
    title: "Xerox Alto",
    description: "El Xerox Alto presentó ventanas, iconos y un puntero controlado por ratón, conceptos tomados y popularizados por Apple con el Macintosh en 1984.",
    image:"https://i0.wp.com/clipset.com/wp-content/uploads/2016/03/XeroxStar.png?resize=552%2C405&ssl=1"
  },
  {
    year: 1993,
    title: "Navegador Mosaic",
    description: "Primer navegador en integrar imágenes y texto en una misma página, ofreciendo una experiencia gráfica de navegación web.",
    image:"https://i.blogs.es/5bfa64/mosaic3/1366_2000.jpg"
  },
  {
    year: 2007,
    title: "iPhone",
    description: "Apple lanzó el iPhone, popularizando pantallas multitáctiles y transformando la industria de dispositivos móviles.",
    image:"https://nypost.com/wp-content/uploads/sites/2/2023/01/newspress-collage-25288920-1673246153542.jpg?quality=75&strip=all&1673228410"
  },
  {
    year: 2011,
    title: "Siri",
    description: "El primer asistente de voz integrado en smartphones, marcando el inicio de las interfaces de voz modernas.",
    image:"https://ca-times.brightspotcdn.com/dims4/default/7926050/2147483647/strip/true/crop/2048x1590+0+0/resize/1200x932!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2Fd3%2Fb2%2F3e15df5a3c222bbe4454626a7b9b%2Fla-1465604174-snap-photo"
  },
  {
    year: 2014,
    title: "HTML5",
    description: "Introducción de HTML5, que permitió el desarrollo de interfaces web dinámicas y adaptables sin necesidad de Flash.",
    image:"https://i.extremetech.com/imagery/content-types/03Ax7Sk6hXltIIdrI3L74jH/hero-image.fit_lim.v1678673321.jpg"
  },
  {
    year: 2016,
    title: "Realidad Virtual y Aumentada",
    description: "Con Oculus Rift y Pokémon GO, la realidad virtual y aumentada se consolidaron como formas inmersivas de interacción.",
    image:"https://20963350.fs1.hubspotusercontent-na1.net/hubfs/20963350/Apple-Vision-Pro--caracter%C3%ADsticas-y-precio-del-nuevo-visor-de-Apple.jpg"
  },
  {
    year: 2022,
    title: "OpenAI GPT-3",
    description: "Lanzamiento de GPT-3, un modelo de lenguaje avanzado que transformó las interfaces conversacionales.",
    image:"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZfRnBiCVspvrMlV-yNNORHBdpqq7q_NSoYg&s"
  },
];



export function Timeline() {
  return (
    <div className="relative w-full h-screen flex items-center">
      {/* Línea de tiempo */}
      <motion.div
        className="absolute left-0 right-0 h-1 bg-gray-300 top-[40%] -translate-y-1/2"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 1.5, ease: "easeInOut" }}
      />
      <div className="flex justify-evenly w-full px-10 sm:px-20">
        {timelineData.map((item, index) => (
          <TimelinePoint
            key={item.year}
            year={item.year}
            title={item.title}
            description={item.description}
            image={item.image}
          />
        ))}
      </div>
    </div>
  );
}