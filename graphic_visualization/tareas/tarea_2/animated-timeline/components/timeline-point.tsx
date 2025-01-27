import { useState } from "react";
import { motion } from "framer-motion";
import { CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface TimelinePointProps {
  year: number;
  title: string;
  description: string;
  image?: string;
}

export function TimelinePoint({ year, title, description, image }: TimelinePointProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div className="relative flex flex-col items-center">
      {/* Año arriba del punto */}
      <div
        style={{
          position: "absolute",
          top: "-8.5rem", // Posiciona el año justo encima del punto y la línea
          left: "50%",
          transform: "translateX(-50%)", // Centra el año horizontalmente
        }}
        className="text-sm font-medium"
      >
        {year}
      </div>

      {/* Punto que se transforma en el Card */}
      <motion.div
        className="relative z-10"
        onHoverStart={() => setIsHovered(true)}
        onHoverEnd={() => setIsHovered(false)}
        initial={{ width: "1rem", height: "1rem", borderRadius: "50%" }}
        animate={
          isHovered
            ? { width: "16rem", height: "20rem", borderRadius: "1rem", backgroundColor: "#fff" }
            : { width: "1rem", height: "1rem", borderRadius: "50%", backgroundColor: "#3f3e3d" }
        }
        transition={{ duration: 0.3 }}
        style={{
          position: "absolute",
          top: "-6.5rem", // Alinea el punto con la línea del tiempo
          left: "50%",
          transform: "translateX(-50%)", // Centra el punto horizontalmente
        }}
      >
        {/* Contenido del Card dentro del punto transformado */}
        {isHovered && (
          <div className="p-4 text-left">
            {/* Imagen opcional */}
            {image && (
              <img
                src={image}
                alt={`${title} related`}
                className="w-full h-32 object-cover rounded-lg mb-4"
              />
            )}
            {/* Título */}
            <CardHeader>
              <CardTitle className="text-black text-lg font-bold">{title}</CardTitle>
            </CardHeader>
            {/* Descripción */}
            <CardContent>
              <p className="text-gray-700 text-sm">{description}</p>
            </CardContent>
          </div>
        )}
      </motion.div>
    </div>
  );
}
